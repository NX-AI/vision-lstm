import torch

from ksuit.callbacks.base.callback_base import CallbackBase


class NanHookMonitorCallback(CallbackBase):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose

    def _before_training(self, model, **kwargs):
        for name, module in model.named_modules():
            module.register_forward_hook(self.NanMonitorHook(name, logger=self.logger))

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
