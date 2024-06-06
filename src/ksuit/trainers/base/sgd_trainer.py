import inspect
import logging
import os

import kappaprofiler as kp
import torch
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ksuit.callbacks.base.callback_base import CallbackBase
from ksuit.callbacks.base.periodic_callback import PeriodicCallback
from ksuit.callbacks.default_callbacks import (
    CopyPreviousConfigCallback,
    CopyPreviousSummaryCallback,
    DatasetStatsCallback,
    EtaCallback,
    FreezerCallback,
    LrCallback,
    OnlineLossCallback,
    ParamCountCallback,
    ProgressCallback,
    TrainTimeCallback,
)
from ksuit.distributed import (
    is_distributed,
    get_world_size,
    is_managed,
    is_rank0,
    get_num_nodes,
    all_gather_nograd,
)
from ksuit.factory import MasterFactory
from ksuit.initializers import ResumeInitializer
from ksuit.models import ModelBase
from ksuit.providers import (
    ConfigProviderBase,
    NoopConfigProvider,
    PathProvider,
    SummaryProviderBase,
    NoopSummaryProvider,
    MetricPropertyProvider,
)
from ksuit.utils.amp_utils import get_supported_precision, get_grad_scaler_and_autocast_context
from ksuit.utils.checkpoint import Checkpoint
from ksuit.utils.data_container import DataContainer
from ksuit.utils.math_utils import get_powers_of_two, is_power_of_two
from ksuit.utils.model_utils import get_nograd_paramnames, get_trainable_param_count
from ksuit.utils.param_checking import check_all_none
from ksuit.utils.update_counter import UpdateCounter


class SgdTrainer:
    def __init__(
            self,
            data_container: DataContainer,
            device: str,
            precision,
            effective_batch_size: int,
            main_sampler: dict = None,
            max_epochs=None,
            max_updates=None,
            max_samples=None,
            start_at_epoch=None,
            stop_at_epoch=None,
            stop_at_update=None,
            stop_at_sample=None,
            add_default_callbacks: bool = True,
            add_trainer_callbacks: bool = True,
            callbacks: list = None,
            backup_precision: str = None,
            log_every_n_epochs=None,
            log_every_n_updates=None,
            log_every_n_samples=None,
            track_every_n_epochs=None,
            track_every_n_updates=None,
            track_every_n_samples=None,
            early_stoppers=None,
            initializer: ResumeInitializer = None,
            disable_gradient_accumulation: bool = False,
            force_gradient_accumulation: bool = False,
            max_batch_size: int = None,
            sync_batchnorm: bool = True,
            skip_nan_loss: bool = False,
            # find_unused_params should not be set to true if it is not needed (to avoid overhead)
            # but sometimes it is required (e.g. when dynamically freezing/unfreezing parameters)
            # when find_unused_params setting static_graph to true can bring speedup
            find_unused_params: bool = False,
            static_graph: bool = False,
            use_torch_compile: bool = False,
            # providers
            config_provider: ConfigProviderBase = None,
            summary_provider: SummaryProviderBase = None,
            path_provider: PathProvider = None,
            metric_property_provider: MetricPropertyProvider = None,
    ):
        self.logger = logging.getLogger(type(self).__name__)
        self.data_container = data_container
        self.config_provider = config_provider or NoopConfigProvider()
        self.summary_provider = summary_provider or NoopSummaryProvider()
        self.path_provider = path_provider
        self.metric_property_provider = metric_property_provider

        self.device: torch.device = torch.device(device)
        self.effective_batch_size = effective_batch_size
        self.end_checkpoint = Checkpoint(max_epochs, max_updates, max_samples)
        self.stop_at_epoch = stop_at_epoch
        self.stop_at_update = stop_at_update
        self.stop_at_sample = stop_at_sample
        self.add_default_callbacks = add_default_callbacks
        self.add_trainer_callbacks = add_trainer_callbacks
        self.precision = get_supported_precision(
            desired_precision=precision,
            backup_precision=backup_precision,
            device=self.device,
        )
        self.logger.info(f"using precision: {self.precision} (desired={precision} backup={backup_precision})")
        self.grad_scaler, self.autocast_context = get_grad_scaler_and_autocast_context(self.precision, self.device)
        self.log_every_n_epochs = log_every_n_epochs
        self.log_every_n_updates = log_every_n_updates
        self.log_every_n_samples = log_every_n_samples
        # by default track every 50 updates
        if check_all_none(track_every_n_epochs, track_every_n_updates, track_every_n_samples):
            track_every_n_updates = 50
        self.track_every_n_epochs = track_every_n_epochs
        self.track_every_n_updates = track_every_n_updates
        self.track_every_n_samples = track_every_n_samples
        self.early_stoppers = MasterFactory.get("early_stopper").create_list(
            early_stoppers,
            optional_kwargs=dict(metric_property_provider=metric_property_provider),
        )
        self.train_dataset = self.data_container.get_dataset("train", mode=self.dataset_mode)
        self.main_sampler = self.data_container.get_main_sampler(
            train_dataset=self.train_dataset,
            **(main_sampler or {}),
        )
        eff_len = self.main_sampler.effective_length
        assert eff_len >= self.effective_batch_size, f"{eff_len}<{self.effective_batch_size}"
        self.updates_per_epoch = int(eff_len / self.effective_batch_size)
        self.max_batch_size = max_batch_size
        self.disable_gradient_accumulation = disable_gradient_accumulation
        self.force_gradient_accumulation = force_gradient_accumulation
        self.sync_batchnorm = sync_batchnorm
        self.find_unused_params = find_unused_params
        self.static_graph = static_graph
        self.use_torch_compile = use_torch_compile
        self.skip_nan_loss = skip_nan_loss
        self.skip_nan_loss_counter = 0

        self.initializer = MasterFactory.get("initializer").create(
            initializer,
            optional_kwargs=dict(path_provider=self.path_provider),
        )
        if self.initializer is None:
            if start_at_epoch is not None:
                start_epoch = start_at_epoch
                start_update = self.updates_per_epoch * start_epoch
                start_sample = start_update * effective_batch_size
            else:
                start_epoch = 0
                start_update = 0
                start_sample = 0
            self.start_checkpoint = Checkpoint(epoch=start_epoch, update=start_update, sample=start_sample)
        else:
            assert start_at_epoch is None
            self.start_checkpoint = self.initializer.get_start_checkpoint()
        self.update_counter = UpdateCounter(
            start_checkpoint=self.start_checkpoint,
            end_checkpoint=self.end_checkpoint,
            updates_per_epoch=self.updates_per_epoch,
            effective_batch_size=self.effective_batch_size,
        )
        self.callbacks = MasterFactory.get("callback").create_list(
            callbacks,
            optional_kwargs=dict(
                data_container=self.data_container,
                config_provider=self.config_provider,
                summary_provider=self.summary_provider,
                path_provider=self.path_provider,
                metric_property_provider=self.metric_property_provider,
                update_counter=self.update_counter,
            ),
        )

        # check that children only override their implementation methods
        assert type(self).train == SgdTrainer.train
        assert type(self).wrap_model == SgdTrainer.wrap_model

    @property
    def input_shape(self):
        dataset = self.data_container.get_dataset("train", mode="x")
        dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=dataset.collator)
        sample = next(iter(dataloader))["x"]
        if torch.is_tensor(sample):
            input_shape = sample[0].shape
        elif isinstance(sample, list):
            input_shape = sample[0][0].shape
        else:
            raise NotImplementedError
        self.logger.info(f"input_shape: {tuple(input_shape)}")
        return tuple(input_shape)

    def get_all_callbacks(self, model=None):
        # no default/trainer callbacks needed for eval runs
        if self.end_checkpoint.epoch == 0 or self.end_checkpoint.update == 0 or self.end_checkpoint.sample == 0:
            return self.callbacks

        # add default/trainer callbacks
        callbacks = []
        if self.add_default_callbacks:
            callbacks += self.get_default_callbacks()
        if self.add_trainer_callbacks:
            callbacks += self.get_trainer_callbacks(model=model)
        callbacks += self.callbacks
        return callbacks

    def get_trainer_callbacks(self, model=None):
        return []

    def get_default_callback_kwargs(self):
        return dict(
            data_container=self.data_container,
            config_provider=self.config_provider,
            summary_provider=self.summary_provider,
            path_provider=self.path_provider,
            update_counter=self.update_counter,
        )

    def get_default_callback_intervals(self):
        return dict(
            every_n_epochs=self.log_every_n_epochs,
            every_n_updates=self.log_every_n_updates,
            every_n_samples=self.log_every_n_samples,
        )

    def get_default_callbacks(self):
        default_kwargs = self.get_default_callback_kwargs()
        default_intervals = self.get_default_callback_intervals()
        # statistic callbacks
        default_callbacks = [
            DatasetStatsCallback(**default_kwargs),
            ParamCountCallback(**default_kwargs),
        ]
        # copy config/summary/entries
        default_callbacks += [
            CopyPreviousConfigCallback(**default_kwargs),
            CopyPreviousSummaryCallback(**default_kwargs),
        ]

        # add default training loggers (not needed for eval runs)
        if not self.update_counter.is_finished:
            # periodic callbacks
            default_callbacks += [
                ProgressCallback(**default_kwargs, **default_intervals),
                TrainTimeCallback(**default_kwargs, **default_intervals),
                OnlineLossCallback(**default_kwargs, **default_intervals, verbose=True),
            ]

            # EtaCallback is pointless in managed runs
            # - managed runs don't have an interactive console
            if not is_managed() and is_rank0():
                default_callbacks = [EtaCallback(**default_kwargs, **default_intervals)] + default_callbacks

            track_kwargs = dict(
                every_n_epochs=self.track_every_n_epochs,
                every_n_updates=self.track_every_n_updates,
                every_n_samples=self.track_every_n_samples,
            )
            default_callbacks += [
                LrCallback(**default_kwargs, **track_kwargs),
                FreezerCallback(**default_kwargs, **track_kwargs),
                OnlineLossCallback(**default_kwargs, **track_kwargs, verbose=False)
            ]

        for callback in default_callbacks:
            self.logger.info(f"added default {callback}")
        return default_callbacks

    def _automatic_max_batch_size(
            self,
            effective_batch_size_per_device,
            model,
            ddp_model,
    ):
        if self.end_checkpoint.epoch == 0 or self.end_checkpoint.update == 0 or self.end_checkpoint.sample == 0:
            self.logger.info(f"eval run -> disable gradient accumulation")
            return effective_batch_size_per_device
        if str(model.device) == "cpu":
            self.logger.info(f"device is cpu -> disable gradient accumulation")
            return effective_batch_size_per_device
        # batchsizes that are not a power of two are not supported
        if not is_power_of_two(effective_batch_size_per_device):
            self.logger.info(f"batchsize is not a power of two -> disable gradient accumulation")
            return effective_batch_size_per_device

        # backup state_dict (state_dict doesn't clone tensors -> call .clone on every tensor in the state dict)
        model_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        if isinstance(model, ModelBase):
            optim_state_dicts = {}
            for name, submodel in model.submodels.items():
                if submodel is None or submodel.optim is None:
                    continue
                sd = submodel.optim.state_dict()
                cloned = {}
                for key in sd.keys():
                    if key == "state":
                        cloned["state"] = {
                            idx_key: {k: v.clone() if v is not None else v for k, v in idx_dict.items()}
                            for idx_key, idx_dict in sd["state"].items()
                        }
                    elif key == "param_groups":
                        cloned["param_groups"] = [{k: v for k, v in group.items()} for group in sd["param_groups"]]
                    elif key == "param_idx_to_name":
                        cloned["param_idx_to_name"] = {k: v for k, v in sd["param_idx_to_name"].items()}
                    else:
                        raise NotImplementedError
                optim_state_dicts[name] = cloned
        else:
            optim_state_dicts = None

        # compose batch_sizes to try
        batch_sizes = get_powers_of_two(2, effective_batch_size_per_device)

        # make a train_step with decreasing batch_sizes (faster when batchsize is actually correct)
        sample = self.train_dataset[0]
        max_batch_size = 1
        for batch_size in reversed(batch_sizes):
            logging.info(f"trying batch_size {batch_size}")

            # compose batch by repeating samples
            # NOTE: collator needs to be called seperately (e.g. for creating batch_idx when using sparse tensors)
            batch = self.train_dataset.collator([sample] * batch_size)

            # try 2 update steps
            # the first update requires less memory because optim states are None
            # the second update requires the full memory consumption
            try:
                for _ in range(2):
                    # optim step is only taken on (iter_step + 1) % accumulation_steps == 0
                    self.update(
                        model=model,
                        ddp_model=ddp_model,
                        batch=batch,
                        iter_step=0,
                        accumulation_steps=1,
                    ),
                max_batch_size = batch_size
                break
            except RuntimeError as e:
                if not str(e).startswith("CUDA out of memory"):
                    raise e
                if isinstance(model, ModelBase):
                    model.clear_buffers()

        # restore state_dict
        model.load_state_dict(model_state_dict)
        if isinstance(model, ModelBase):
            for name, submodel in model.submodels.items():
                if submodel is None or submodel.optim is None:
                    continue
                submodel.optim.load_state_dict(optim_state_dicts[name])
            # clear buffers if models track something during the forward pass --> e.g. NnclrQueue
            model.clear_buffers()
        return max_batch_size

    def _calculate_batch_size_and_accumulation_steps(self, model, ddp_model):
        self.logger.info(
            f"calculating batch_size and accumulation_steps "
            f"(effective_batch_size={self.effective_batch_size})"
        )
        # calculate effective_batch_size_per_device
        world_size = get_world_size()
        assert self.effective_batch_size % world_size == 0, \
            f"effective_batch_size ({self.effective_batch_size}) needs to be multiple of world_size ({world_size})"
        effective_batch_size_per_device = int(self.effective_batch_size / world_size)
        if self.end_checkpoint.update == 0:
            self.logger.info("eval run -> no automatic batchsize")
            return effective_batch_size_per_device, 1
        if model.is_batch_size_dependent:
            if self.force_gradient_accumulation:
                self.logger.info("model is batch_size dependent but gradient accumulation is forced")
                assert not self.disable_gradient_accumulation
            else:
                self.logger.info("model is batch_size dependent -> disabled gradient accumulation")
                return effective_batch_size_per_device, 1
        if self.disable_gradient_accumulation:
            assert not self.force_gradient_accumulation
            self.logger.info(f"gradient accumulation disabled")
            return effective_batch_size_per_device, 1
        if get_num_nodes() > 1 and self.max_batch_size is None:
            self.logger.info(f"found multi-node setting -> disable automatic batchsize (occasionally hangs)")
            return effective_batch_size_per_device, 1
        if self.use_torch_compile and self.max_batch_size is None:
            self.logger.info("torch.compile is used -> automatic batchsize not supported")
            return effective_batch_size_per_device, 1

        self.logger.info(f"effective_batch_size: {self.effective_batch_size}")
        if is_distributed():
            self.logger.info(f"effective_batch_size_per_device: {effective_batch_size_per_device}")
            self.logger.info(f"world_size: {get_world_size()}")

        if self.max_batch_size is None:
            # calculate max_batch_size
            self.logger.info("calculating automatic max_batch_size")
            max_batch_size = self._automatic_max_batch_size(
                effective_batch_size_per_device=effective_batch_size_per_device,
                model=model,
                ddp_model=ddp_model,
            )
            self.logger.info(f"automatic max_batch_size: {max_batch_size}")
            if is_distributed():
                # check if all devices have the same max_batch_size
                max_batch_sizes = all_gather_nograd(max_batch_size).tolist()
                assert all(max_batch_size == mbs for mbs in max_batch_sizes)
        else:
            assert self.max_batch_size % world_size == 0, \
                f"max_batch_size ({self.max_batch_size}) needs to be multiple of world_size ({world_size})"
            max_batch_size = int(self.max_batch_size / world_size)
            self.logger.info(f"using provided max_batch_size {self.max_batch_size} ({max_batch_size} per device)")

        # calculate batch_size and accumulation_steps
        if effective_batch_size_per_device <= max_batch_size:
            # fits into memory
            batch_size = effective_batch_size_per_device
            accumulation_steps = 1
        else:
            # multiple accumulation steps
            assert effective_batch_size_per_device % max_batch_size == 0, \
                "effective_batch_size_per_device needs to be multiple of max_batch_size"
            accumulation_steps = int(effective_batch_size_per_device / max_batch_size)
            batch_size = int(effective_batch_size_per_device / accumulation_steps)
        self.logger.info(f"batch_size: {batch_size}")
        self.logger.info(f"accumulation_steps: {accumulation_steps}")
        return batch_size, accumulation_steps

    def state_dict(self):
        callback_state_dicts = [callback.state_dict() for callback in self.callbacks]
        state_dict = dict(
            epoch=self.update_counter.cur_checkpoint.epoch,
            update=self.update_counter.cur_checkpoint.update,
            sample=self.update_counter.cur_checkpoint.sample,
            callback_state_dicts=callback_state_dicts,
        )
        if isinstance(self.grad_scaler, GradScaler):
            state_dict["grad_scaler"] = self.grad_scaler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        # shallow copy
        state_dict = {k: v for k, v in state_dict.items()}

        # load callback state_dicts
        callback_state_dicts = state_dict.pop("callback_state_dicts")
        for callback, sd in zip(self.callbacks, callback_state_dicts):
            callback.load_state_dict(sd)

        # load grad_scaler
        grad_scaler_state_dict = state_dict.pop("grad_scaler", None)
        if isinstance(self.grad_scaler, GradScaler):
            if grad_scaler_state_dict is None:
                self.logger.warning(
                    f"trainer checkpoint has no grad_scaler but current trainer uses {self.precision} precision"
                )
            else:
                self.grad_scaler.load_state_dict(grad_scaler_state_dict)

    @property
    def lr_scale_factor(self):
        return self.effective_batch_size

    def _prepare_model(self, model):
        model = model.to(self.device)
        model.initialize(lr_scale_factor=self.lr_scale_factor)
        self.apply_resume_initializer(model)
        return model

    def apply_resume_initializer(self, model):
        # initialize model to state where it was resumed from
        if self.initializer is not None:
            self.logger.info("------------------")
            self.logger.info("loading trainer/model state for resuming")
            assert isinstance(self.initializer, ResumeInitializer)
            self.logger.info(
                f"loading state from checkpoint {self.initializer.stage_id}/"
                f"{self.initializer.stage_name}/{self.initializer.checkpoint}"
            )
            self.initializer.init_trainer(self)
            self.initializer.init_weights(model)
            self.initializer.init_optim(model)
            self.initializer.init_callbacks(self.callbacks, model=model)

    def get_data_loader(self, periodic_callbacks, batch_size):
        self.logger.info(f"initializing dataloader")
        configs = []
        for c in periodic_callbacks:
            cur_configs, _ = c.register_sampler_configs(self)
            for cur_config in cur_configs:
                if hasattr(cur_config.sampler, "data_source"):
                    dataset_mode = cur_config.sampler.data_source.mode
                elif hasattr(cur_config.sampler, "dataset"):
                    dataset_mode = cur_config.sampler.dataset.mode
                else:
                    dataset_mode = "unknown"
                self.logger.info(f"{c} registered {cur_config} dataset_mode='{dataset_mode}'")
            configs += cur_configs
        kwargs = {}
        if self.start_checkpoint.epoch != 0:
            kwargs["start_epoch"] = self.start_checkpoint.epoch
        return self.data_container.get_data_loader(
            main_sampler=self.main_sampler,
            main_collator=self.train_dataset.collator,
            batch_size=batch_size,
            epochs=self.end_checkpoint.epoch,
            updates=self.end_checkpoint.update,
            samples=self.end_checkpoint.sample,
            configs=configs,
            **kwargs,
        )

    def wrap_model(self, model):
        assert model.is_initialized, "Model needs to be initialized before DDP wrapping as DPP broadcasts params"
        model = self._wrap_model(model=model)
        trainer_model = self.get_trainer_model(model)
        ddp_model = self.wrap_ddp(trainer_model)
        ddp_model = self.wrap_compile(ddp_model)
        return model, trainer_model, ddp_model

    def get_trainer_model(self, model):
        raise NotImplementedError

    @staticmethod
    def _wrap_model(model):
        return model

    def wrap_ddp(self, trainer_model):
        if is_distributed():
            if get_trainable_param_count(trainer_model) > 0:
                if self.find_unused_params:
                    self.logger.info(f"using find_unused_params=True")
                if self.static_graph:
                    self.logger.info(f"using static_graph=True")
                trainer_model = DistributedDataParallel(
                    trainer_model,
                    find_unused_parameters=self.find_unused_params,
                    static_graph=self.static_graph,
                )
            else:
                # DDP broadcasts weights from rank0 to other ranks but raises an error if no param requires grad
                # workaround: temporarily unfreeze one parameter if all parameters are frozen to broadcast weights
                self.logger.info(f"not wrapping into DDP (no trainable parameters) -> only broadcast parameters")
                first_param = next(trainer_model.parameters())
                first_param.requires_grad = True
                DistributedDataParallel(trainer_model)
                first_param.requires_grad = False
            if trainer_model.device != torch.device("cpu") and self.sync_batchnorm:
                self.logger.info(f"replacing BatchNorm layers with SyncBatchNorm")
                trainer_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(trainer_model)
        return trainer_model

    def wrap_compile(self, ddp_model):
        if not self.use_torch_compile or os.name == "nt":
            self.logger.info(f"torch.compile not used (use_torch_compile == False)")
            return ddp_model
        if is_distributed():
            if self.static_graph:
                self.logger.info(f"torch.compile static_graph=True is not supported -> disable torch.compile")
                return ddp_model
        self.logger.info(f"wrapping model with torch.compile")
        return torch.compile(ddp_model)

    def before_training(self, model):
        pass

    @kp.profile
    def train(self, model, callbacks=None):
        model = self._prepare_model(model)
        callbacks = callbacks or self.get_all_callbacks(model=model)
        periodic_callbacks = [callback for callback in callbacks if isinstance(callback, PeriodicCallback)]

        self.before_training(model)
        model, trainer_model, ddp_model = self.wrap_model(model)
        batch_size, accumulation_steps, train_batches_per_epoch = self._prepare_batch_size(model, ddp_model)
        assert trainer_model.model == model
        # TODO model is moved to GPU seperately from trainer_model because of initializers
        #  -> trainer_model should be moved all at once
        #  -> wrap_model requires model to be on GPU because of check for SyncBatchNorm
        trainer_model = trainer_model.to(model.device)

        data_loader = self.get_data_loader(periodic_callbacks=periodic_callbacks, batch_size=batch_size)
        self.call_before_training(trainer_model=trainer_model, batch_size=batch_size, callbacks=callbacks)
        self._train(
            model=model,
            trainer_model=trainer_model,
            ddp_model=ddp_model,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            data_loader=data_loader,
            train_batches_per_epoch=train_batches_per_epoch,
            periodic_callbacks=periodic_callbacks,
        )
        self.call_after_training(trainer_model=trainer_model, callbacks=callbacks)

    def _train(
            self,
            model,
            trainer_model,
            ddp_model,
            batch_size,
            accumulation_steps,
            data_loader,
            train_batches_per_epoch,
            periodic_callbacks
    ):
        self.logger.info("------------------")
        self.logger.info(f"START TRAINING")

        self.logger.info("initializing dataloader workers")
        with kp.named_profile("iterator"):
            data_iter = iter(data_loader)
        self.logger.info("initialized dataloader workers")

        if self.update_counter.is_finished:
            if not model.is_frozen:
                self.logger.warning("model has optimizer which is not used for evaluation")
            # eval run
            for callback in periodic_callbacks:
                callback.after_epoch(
                    update_counter=self.update_counter,
                    effective_batch_size=self.effective_batch_size,
                    batch_size=batch_size,
                    trainer=self,
                    model=model,
                    trainer_model=trainer_model,
                    data_iter=data_iter,
                )
            for callback in periodic_callbacks:
                callback.after_update(
                    update_counter=self.update_counter,
                    effective_batch_size=self.effective_batch_size,
                    batch_size=batch_size,
                    trainer=self,
                    model=model,
                    trainer_model=trainer_model,
                    data_iter=data_iter,
                )
            CallbackBase.flush()
        else:
            # train run
            is_first_update = True
            while True:
                iter_step = -1
                data_time = 0.
                update_time = 0.
                while True:
                    # check end of epoch
                    remaining_batches = train_batches_per_epoch - (iter_step + 1)
                    if remaining_batches < accumulation_steps:
                        # InterleavedSampler already have the next batches preloaded -> skip them
                        for _ in range(remaining_batches):
                            _ = next(data_iter)
                        break
                    is_last_update_in_epoch = remaining_batches - accumulation_steps < accumulation_steps
                    for callback in periodic_callbacks:
                        callback.before_every_update(update_counter=self.update_counter, model=model)
                    for _ in range(accumulation_steps):
                        # load next batch
                        with kp.named_profile("data_loading"):
                            batch = next(data_iter)
                            iter_step += 1
                        if iter_step % accumulation_steps == 0:
                            model.optim_schedule_step()
                            data_time = 0.
                            update_time = 0.
                        data_time += kp.profiler.last_node.last_time
                        for callback in periodic_callbacks:
                            callback.before_every_accumulation_step(model=model, batch=batch)

                        trainer_model.train()
                        # update contains implicit cuda synchronization points (.detach().cpu(), .item())
                        with kp.named_profile("update"):
                            losses, update_outputs = self.update(
                                batch=batch,
                                iter_step=iter_step,
                                model=model,
                                ddp_model=ddp_model,
                                accumulation_steps=accumulation_steps,
                                is_first_update=is_first_update,
                                periodic_callbacks=periodic_callbacks,
                            )
                        update_time += kp.profiler.last_node.last_time
                        for callback in periodic_callbacks:
                            callback.track_after_accumulation_step(
                                update_counter=self.update_counter,
                                trainer=self,
                                model=model,
                                batch=batch,
                                losses=losses,
                                update_outputs=update_outputs,
                                accumulation_steps=accumulation_steps,
                            )
                        # free references to tensors
                        # noinspection PyUnusedLocal
                        update_outputs = None
                        is_first_update = False

                    # advance counter
                    self.update_counter.add_samples(self.effective_batch_size)
                    self.update_counter.next_update()
                    if is_last_update_in_epoch:
                        self.update_counter.next_epoch()

                    trainer_model.eval()
                    times = dict(data_time=data_time, update_time=update_time)
                    for callback in periodic_callbacks:
                        callback.track_after_update_step(
                            update_counter=self.update_counter,
                            trainer=self,
                            model=model,
                            times=times,
                        )
                    for callback in periodic_callbacks:
                        callback.after_update(
                            update_counter=self.update_counter,
                            effective_batch_size=self.effective_batch_size,
                            batch_size=batch_size,
                            trainer=self,
                            model=model,
                            trainer_model=trainer_model,
                            data_iter=data_iter,
                        )
                    # check end of training
                    if self.update_counter.is_finished:
                        # skip preloaded batches after training when accumulation steps > 1
                        if data_loader.batch_sampler.sampler.epochs is not None:
                            for _ in range(remaining_batches - accumulation_steps):
                                _ = next(data_iter)
                        if data_loader.batch_sampler.sampler.samples is not None:
                            total_batches = int(data_loader.batch_sampler.sampler.samples / batch_size)
                            for _ in range(total_batches % accumulation_steps):
                                _ = next(data_iter)
                        break

                    # no end of epoch -> flush logs from call_after_update
                    if not is_last_update_in_epoch:
                        CallbackBase.flush()

                    # check update/sample based early stopping
                    for early_stopper in self.early_stoppers:
                        should_stop_after_update = early_stopper.should_stop_after_update(
                            self.update_counter.cur_checkpoint,
                        )
                        if should_stop_after_update:
                            return
                        should_stop_after_sample = early_stopper.should_stop_after_sample(
                            self.update_counter.cur_checkpoint,
                            effective_batch_size=self.effective_batch_size,
                        )
                        if should_stop_after_sample:
                            return
                    # update based premature stopping
                    if self.stop_at_update is not None:
                        if self.stop_at_update <= self.update_counter.update:
                            self.logger.info(f"reached stop_at_update (={self.stop_at_update}) -> stop training")
                            return
                    # sample based premature stopping
                    if self.stop_at_sample is not None:
                        if self.stop_at_sample <= self.update_counter.sample:
                            self.logger.info(f"reached stop_at_sample (={self.stop_at_sample}) -> stop training")
                            return

                if self.update_counter.is_full_epoch:
                    for callback in periodic_callbacks:
                        callback.after_epoch(
                            update_counter=self.update_counter,
                            effective_batch_size=self.effective_batch_size,
                            batch_size=batch_size,
                            trainer=self,
                            model=model,
                            trainer_model=trainer_model,
                            data_iter=data_iter,
                        )
                    CallbackBase.flush()

                    # check epoch based early stopping
                    for early_stopper in self.early_stoppers:
                        if early_stopper.should_stop_after_epoch(self.update_counter.cur_checkpoint):
                            return
                    # epoch based premature stopping
                    if self.stop_at_epoch is not None:
                        if self.stop_at_epoch <= self.update_counter.epoch:
                            self.logger.info(f"reached stop_at_epoch (={self.stop_at_epoch}) -> stop training")
                            return
                # check end of training
                if self.update_counter.is_finished:
                    break
        # check that data_iter was fully consumed
        unconsumed_data_iter_steps = 0
        try:
            next(data_iter)
            self.logger.error("data_iter was not fully consumed -> checking how many batches remain")
            unconsumed_data_iter_steps = 1
            for _ in range(10):
                next(data_iter)
                unconsumed_data_iter_steps += 1
            raise RuntimeError(f"data_iter was not fully consumed (at least {unconsumed_data_iter_steps} unconsumed)")
        except StopIteration:
            if unconsumed_data_iter_steps > 0:
                raise RuntimeError(f"data_iter was not fully consumed ({unconsumed_data_iter_steps} unconsumed)")

    def update(
            self,
            batch,
            ddp_model,
            model=None,
            training=True,
            forward_kwargs=None,
            periodic_callbacks=None,
            **kwargs,
    ):
        assert training == ddp_model.training
        if training:
            assert forward_kwargs is None
            model.before_accumulation_step()
        forward_kwargs = forward_kwargs or {}
        if inspect.isgeneratorfunction(ddp_model.forward):
            # multiple update steps from same batch by returning via "yield losses, infos"
            generator = ddp_model(batch, **forward_kwargs)
            all_losses = {}
            all_outputs = {}
            i = 0
            while True:
                try:
                    with kp.named_profile_async("forward"):
                        with self.autocast_context:
                            result = next(generator)
                        if len(result) == 3:
                            losses, outputs, retain_graph = result
                        else:
                            losses, outputs = result
                            retain_graph = False
                    cur_total_loss = losses.pop("total")
                    if training:
                        self._step(
                            total_loss=cur_total_loss,
                            model=model,
                            retain_graph=retain_graph,
                            periodic_callbacks=periodic_callbacks,
                            **kwargs,
                        )
                    all_losses.update({k: v.detach() for k, v in losses.items()})
                    all_losses[f"total.{i}"] = cur_total_loss.detach()
                    all_outputs.update(outputs)
                    i += 1
                except StopIteration:
                    break
            all_losses["total"] = sum([v for k, v in all_losses.items() if k.startswith("total.")])
        else:
            # single update step from same batch by returning via "return losses, infos"
            with kp.named_profile_async("forward"):
                with self.autocast_context:
                    losses, outputs = ddp_model(batch, **forward_kwargs)
            if training:
                self._step(
                    total_loss=losses["total"],
                    model=model,
                    periodic_callbacks=periodic_callbacks,
                    **kwargs,
                )
            all_losses = {k: v.detach() for k, v in losses.items()}
            all_outputs = outputs
        return all_losses, all_outputs

    def _step(
            self,
            total_loss,
            model,
            accumulation_steps,
            iter_step,
            is_first_update=False,
            retain_graph=False,
            periodic_callbacks=None,
    ):
        # loss
        total_loss = total_loss / accumulation_steps
        if not model.is_frozen:
            # backward
            for callback in periodic_callbacks or []:
                callback.before_every_backward(
                    update_counter=self.update_counter,
                    effective_batch_size=self.effective_batch_size,
                    trainer=self,
                    model=model,
                )
            with kp.named_profile_async("backward"):
                self.grad_scaler.scale(total_loss).backward(retain_graph=retain_graph)

            # log unused parameters
            if is_first_update:
                unused_param_names = get_nograd_paramnames(model)
                self.logger.info(f"{len(unused_param_names)} unused parameters")
                for name in unused_param_names:
                    self.logger.info(f"- {name}")

            if (iter_step + 1) % accumulation_steps == 0:
                do_step = True
                if self.skip_nan_loss:
                    total_loss = all_gather_nograd(total_loss)
                    if torch.any(torch.isnan(total_loss)):
                        self.logger.info(f"encountered nan loss -> skip (counter: {self.skip_nan_loss})")
                        do_step = False
                        self.skip_nan_loss_counter += 1
                        if self.skip_nan_loss_counter > 100:
                            raise RuntimeError(f"encountered 100 nan losses in a row")
                    else:
                        # reset counter
                        if self.skip_nan_loss_counter > 0:
                            self.logger.info(f"encountered valid loss after {self.skip_nan_loss_counter} nan losses")
                            self.skip_nan_loss_counter = 0

                if do_step:
                    model.optim_step(self.grad_scaler)
                model.optim_zero_grad()

    @property
    def dataset_mode(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        return None

    def _prepare_batch_size(self, model, ddp_model):
        self.logger.info("------------------")
        self.logger.info("PREPARE TRAINER")
        batch_size, accumulation_steps = self._calculate_batch_size_and_accumulation_steps(model, ddp_model)
        if accumulation_steps > 1 and self.end_checkpoint.update is not None:
            raise NotImplementedError(
                "InterleavedSampler counts every batch as update "
                "-> accumulation steps not supported with update-based end_checkpoint"
            )
        # set accumulation steps in model (e.g. for AsyncBatchNorm it is initialized with accumulation_steps=1
        # but needs to be updated once the actual accumulation_steps are known)
        model.set_accumulation_steps(accumulation_steps)
        self.config_provider["trainer/batch_size"] = batch_size
        self.config_provider["trainer/accumulation_steps"] = accumulation_steps
        train_batches_per_epoch = int(
            self.main_sampler.effective_length
            / self.effective_batch_size
            * accumulation_steps
        )
        self.logger.info(
            f"train_batches per epoch: {train_batches_per_epoch} "
            f"(world_size={get_world_size()} batch_size={batch_size})"
        )

        return batch_size, accumulation_steps, train_batches_per_epoch

    def call_before_training(self, trainer_model, batch_size, callbacks):
        self.logger.info("------------------")
        self.logger.info("BEFORE TRAINING")
        trainer_model.eval()
        for c in callbacks:
            c.before_training(
                trainer_model=trainer_model,
                model=trainer_model.model,
                trainer=self,
                update_counter=self.update_counter,
                trainer_batch_size=batch_size,
            )
        self.logger.info("------------------")
        for callback in callbacks:
            self.logger.info(f"{callback}")

    def call_after_training(self, trainer_model, callbacks):
        self.logger.info("------------------")
        self.logger.info("AFTER TRAINING")
        trainer_model.eval()
        for callback in callbacks:
            callback.after_training(model=trainer_model.model, trainer=self, update_counter=self.update_counter)
        CallbackBase.flush()
