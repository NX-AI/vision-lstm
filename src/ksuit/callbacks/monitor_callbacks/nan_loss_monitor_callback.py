import torch
import yaml

from ksuit.callbacks.base.periodic_callback import PeriodicCallback
from ksuit.distributed import get_rank, all_gather_nograd
from ksuit.utils.logging_utils import log_from_all_ranks


class NanLossMonitorCallback(PeriodicCallback):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.input_output_hook = None
        # TODO trainer model should be passed to callbacks
        self.trainer_model = None

    def _before_training(self, trainer_model, **kwargs):
        # TODO trainer model should be passed to callbacks
        self.trainer_model = trainer_model
        # hook into trainer_model to get input (batch) and output (losses)
        self.input_output_hook = self.InputOutputHook()
        trainer_model.register_forward_hook(self.input_output_hook)

    def before_every_backward(self, model, **kwargs):
        # TODO logging from all ranks
        losses, infos = self.input_output_hook.outputs
        total_losses = all_gather_nograd(losses["total"])
        if torch.any(torch.isnan(total_losses)):
            with log_from_all_ranks():
                # write out everything that could be useful for analysis
                out = self.path_provider.stage_output_path / f"nan_loss_monitor_rank{get_rank()}"
                out.mkdir()
                self.logger.info(f"nan loss detected -> storing things to replicate current update in {out.as_posix()}")
                torch.save(self.input_output_hook.inputs, out / "inputs.th")
                torch.save(self.trainer_model.state_dict(), out / "trainer_model.th")
                with open(out / "cur_checkpoint.yaml", "w") as f:
                    yaml.safe_dump(dict(self.trainer_model.trainer.update_counter.cur_checkpoint), f)
                torch.save(losses, out / "losses.th")
                torch.save(infos, out / "infos.th")
                # repeat forward pass
                self.logger.info(f"attempting to repeat update step with nan monitor hooks on every module")
                for name, module in model.named_modules():
                    module.register_forward_hook(self.NanMonitorHook(name, logger=self.logger))
                rep_losses, rep_infos = self.trainer_model(*self.input_output_hook.inputs)
                self.logger.info(f"repeated update step losses: {rep_losses}")
                self.logger.info(f"repeated update step losses: {rep_losses}")
            # this should never happen because hooks should raise an exception once they found the nan/inf value
            raise RuntimeError(f"encountered nan loss but couldnt replicate nan/inf value")
        else:
            # clear hook
            self.input_output_hook.inputs = None
            self.input_output_hook.outputs = None

    class InputOutputHook:
        def __init__(self):
            self.inputs = None
            self.outputs = None

        def __call__(self, module, module_input, module_output):
            if not module.training:
                return
            self.inputs = module_input
            self.outputs = module_output

    class NanMonitorHook:
        def __init__(self, name: str, logger):
            self.name = name
            self.logger = logger

        def _log_and_print(self, msg):
            self.logger.info(msg)
            print(msg)

        def _log(self, module, module_input, module_output, tensor_name, invalid_value):
            self.logger.error("-" * 50)
            self.logger.error(f"encountered {invalid_value} in module {self.name} ({tensor_name})")
            self.logger.info(f"parameters:")
            for name, param in module.named_parameters():
                if param.numel() == 1:
                    self._log_and_print(f"{name}.item(): {param.item()}")
                elif param.numel() > 1:
                    self._log_and_print(f"{name}.abs().max(): {param.abs().max().item()}")
                    self._log_and_print(f"{name}.abs().min(): {param.abs().min().item()}")
                    self._log_and_print(f"{name}.mean(): {param.mean().item()}")
                    self._log_and_print(f"{name}.std(): {param.std().item()}")

            for i in range(len(module_input)):
                if not torch.is_tensor(module_input[i]):
                    continue
                tensor = module_input[i].flatten()
                tensor = tensor[~torch.isnan(tensor)]
                name = f"module_input[{i}]"
                if tensor.numel() == 1:
                    self._log_and_print(f"{name}.item(): {tensor.item()}")
                elif tensor.numel() > 1:
                    self._log_and_print(f"{name}.abs().max(): {tensor.abs().max().item()}")
                    self._log_and_print(f"{name}.abs().min(): {tensor.abs().min().item()}")
                    self._log_and_print(f"{name}.mean(): {tensor.mean().item()}")
                    self._log_and_print(f"{name}.std(): {tensor.std().item()}")

            for i in range(len(module_output)):
                tensor = module_output[i].flatten()
                tensor = tensor[~torch.isnan(tensor)]
                name = f"module_output[{i}]"
                if tensor.numel() == 1:
                    self._log_and_print(f"{name}.item(): {tensor.item()}")
                elif tensor.numel() > 1:
                    self._log_and_print(f"{name}.abs().max(): {tensor.abs().max().item()}")
                    self._log_and_print(f"{name}.abs().min(): {tensor.abs().min().item()}")
                    self._log_and_print(f"{name}.mean(): {tensor.mean().item()}")
                    self._log_and_print(f"{name}.std(): {tensor.std().item()}")
            self.logger.error(f"encountered {invalid_value} in module {self.name} ({tensor_name})")
            # TODO somthing like dist.kill_process_group
            raise RuntimeError(f"encountered {invalid_value} in module {self.name} ({tensor_name})")

        def __call__(self, module, module_input, module_output):
            assert isinstance(module_input, tuple)
            for i in range(len(module_input)):
                if torch.is_tensor(module_input[i]):
                    if torch.any(torch.isnan(module_input[i])):
                        self._log(
                            module=module,
                            module_input=module_input,
                            module_output=module_output,
                            tensor_name=f"module_input[{i}]",
                            invalid_value="nan",
                        )
                    if torch.any(torch.isinf(module_input[i])):
                        self._log(
                            module=module,
                            module_input=module_input,
                            module_output=module_output,
                            tensor_name=f"module_input[{i}]",
                            invalid_value="inf",
                        )

            if isinstance(module_output, tuple):
                for i in range(len(module_output)):
                    if module_output[i] is None:
                        continue
                    assert torch.is_tensor(module_output[i])
                    if torch.any(torch.isnan(module_output[i])):
                        self._log(
                            module=module,
                            module_input=module_input,
                            module_output=module_output,
                            tensor_name=f"module_output[{i}]",
                            invalid_value="nan",
                        )
                    if torch.any(torch.isinf(module_output[i])):
                        self._log(
                            module=module,
                            module_input=module_input,
                            module_output=module_output,
                            tensor_name=f"module_output[{i}]",
                            invalid_value="inf",
                        )
