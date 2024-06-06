from collections import defaultdict, deque

import torch

from ksuit.callbacks.base.callback_base import CallbackBase


class GradientSpikeMonitorCallback(CallbackBase):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose

    def _before_training(self, model, **kwargs):
        for name, module in model.named_modules():
            module.register_full_backward_hook(self.NanMonitorHook(name, logger=self.logger))

    class NanMonitorHook:
        def __init__(self, name: str, logger):
            self.name = name
            self.logger = logger
            self.num_history_steps = 3
            self.history = defaultdict(lambda: deque([], self.num_history_steps))

        def _log_and_print(self, msg):
            self.logger.info(msg)
            print(msg)

        def _log(self, module, module_input, module_output, tensor_name):
            self.logger.error(f"encountered high gradient magnitude in module {self.name} ({tensor_name})")
            self.logger.info(f"parameters:")
            for name, param in module.named_parameters():
                self._log_and_print(f"{name}.abs().max(): {param.abs().max().item()}")
                self._log_and_print(f"{name}.abs().min(): {param.abs().min().item()}")
                self._log_and_print(f"{name}.mean(): {param.mean().item()}")
                self._log_and_print(f"{name}.std(): {param.std().item()}")

            for i in range(len(module_input)):
                tensor = module_input[i].flatten()
                tensor = tensor[~torch.isnan(tensor)]
                name = f"module_input[{i}]"
                self._log_and_print(f"{name}.abs().max(): {tensor.abs().max().item()}")
                self._log_and_print(f"{name}.abs().min(): {tensor.abs().min().item()}")
                self._log_and_print(f"{name}.mean(): {tensor.mean().item()}")
                self._log_and_print(f"{name}.std(): {tensor.std().item()}")

            for i in range(len(module_output)):
                tensor = module_output[i].flatten()
                tensor = tensor[~torch.isnan(tensor)]
                name = f"module_output[{i}]"
                self._log_and_print(f"{name}.abs().max(): {tensor.abs().max().item()}")
                self._log_and_print(f"{name}.abs().min(): {tensor.abs().min().item()}")
                self._log_and_print(f"{name}.mean(): {tensor.mean().item()}")
                self._log_and_print(f"{name}.std(): {tensor.std().item()}")

            self.logger.error(f"encountered high gradient magnitude in module {self.name} ({tensor_name})")
            exit(0)

        def __call__(self, module, grad_input, grad_output):
            assert isinstance(grad_input, tuple)
            for i in range(len(grad_input)):
                if grad_input[i] is None:
                    continue
                if grad_input[i].dtype == torch.long:
                    continue
                key = f"grad_input[{i}]"
                history = self.history[key]
                mag = grad_input[i].abs().mean()
                # check burn-in
                if len(history) < self.num_history_steps:
                    history.append(mag)
                    continue
                # check for outliers
                history_mean = torch.stack(list(history)).mean()
                if mag > history_mean * 2:
                    self._log(
                        module=module,
                        module_input=grad_input,
                        module_output=grad_output,
                        tensor_name=key
                    )
                history.append(mag)

            assert isinstance(grad_output, tuple)
            for i in range(len(grad_output)):
                if grad_output[i] is None:
                    continue
                key = f"grad_output[{i}]"
                history = self.history[key]
                mag = grad_output[i].abs().mean()
                # check burn-in
                if len(history) < self.num_history_steps:
                    history.append(mag)
                    continue
                # check for outliers
                history_mean = torch.stack(list(history)).mean()
                if mag > history_mean * 2:
                    self._log(
                        module=module,
                        module_input=grad_input,
                        module_output=grad_output,
                        tensor_name=key
                    )
                history.append(mag)
