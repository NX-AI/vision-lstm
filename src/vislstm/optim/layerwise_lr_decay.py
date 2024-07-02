from ksuit.optim.param_group_modifiers import ParamGroupModifierBase


class LayerwiseLrDecay(ParamGroupModifierBase):
    def __init__(self, layerwise_lr_decay, mlp_is_layer=False, group_two_blocks=False):
        self.layerwise_lr_decay = layerwise_lr_decay
        self.mlp_is_layer = mlp_is_layer
        self.group_two_blocks = group_two_blocks

    def get_properties(self, model, name, param):
        # adapted from BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        # this will split the model into len(blocks) + 2 "layers"
        # stem (patch_embed, cls_token, pos_embed) -> blocks -> last norm
        # this means that the last block will already be decayed
        if hasattr(model, "blocks"):
            assert not self.group_two_blocks
            if self.mlp_is_layer:
                num_layers = len(model.blocks) * 2 + 1
            else:
                num_layers = len(model.blocks) + 1
        elif hasattr(model, "model"):
            # e.g. torch_hub_model
            if hasattr(model.model, "blocks"):
                assert not self.group_two_blocks
                if self.mlp_is_layer:
                    num_layers = len(model.model.blocks) * 2 + 1
                else:
                    num_layers = len(model.model.blocks) + 1
            elif hasattr(model.model, "layers"):
                assert not self.mlp_is_layer
                # vision-mamba
                if self.group_two_blocks:
                    assert len(model.model.layers) % 2 == 0
                    num_layers = len(model.model.layers) // 2 + 1
                else:
                    num_layers = len(model.model.layers) + 1
            else:
                raise NotImplementedError
            if name.startswith("model."):
                name = name[len("model."):]
        elif hasattr(model, "xlstm"):
            assert not self.mlp_is_layer
            assert hasattr(model.xlstm, "blocks")
            if self.group_two_blocks:
                assert len(model.xlstm.blocks) % 2 == 0
                num_layers = len(model.xlstm.blocks) // 2 + 1
            else:
                num_layers = len(model.xlstm.blocks) + 1
            if name.startswith("xlstm."):
                name = name[len("xlstm."):]
        else:
            raise NotImplementedError
        scales = list(self.layerwise_lr_decay ** (num_layers - i) for i in range(num_layers))

        if (
                name.startswith("cls_token")
                or name.startswith("pos_embed")
                or name == "mask_token"
        ):
            return dict(lr_scale=scales[0])
        if name.startswith("patch_embed"):
            return dict(lr_scale=scales[0])
        elif name.startswith("block") or name.startswith("layers"):
            if self.mlp_is_layer:
                layer = int(name.split('.')[1]) * 2 + 1
                if "norm2" in name or "mlp" in name or "ls2" in name:
                    layer += 1
            else:
                layer = int(name.split('.')[1]) + 1
            if self.group_two_blocks:
                layer = (layer - 1) // 2 + 1
            return dict(lr_scale=scales[layer])
        elif name.startswith("norm.") or name.startswith("head.") or name.startswith("post_blocks_norm."):
            # last norm is not scaled (i.e. original learning rate)
            return {}
        elif name.startswith("norm_f."):
            # vim norm
            return {}
        else:
            raise NotImplementedError

    def __str__(self):
        return (
            f"{type(self).__name__}("
            f"layerwise_lr_decay={self.layerwise_lr_decay},"
            f")"
        )
