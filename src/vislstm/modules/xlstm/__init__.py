from .blocks.mlstm.block import mLSTMBlock, mLSTMBlockConfig, mLSTMLayerConfig
from .blocks.xlstm_block import xLSTMBlock, xLSTMBlockConfig
from .xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig


def create_block_stack(
        dim,
        depth,
        context_length,
        bidirectional=False,
        quaddirectional=False,
        layerscale=None,
        alternation=None,
        sharedirs=False,
        dropout_rate=0.,
        proj_factor=2.0,
        add_post_blocks_norm=True,
):
    return xLSTMBlockStack(
        xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=4,
                    bidirectional=bidirectional,
                    quaddirectional=quaddirectional,
                    layerscale=layerscale,
                    alternation=alternation,
                    sharedirs=sharedirs,
                    proj_factor=proj_factor,
                ),
            ),
            # slstm_block=sLSTMBlockConfig(
            #     slstm=sLSTMLayerConfig(
            #         backend="vanilla",
            #         num_heads=4,
            #         conv1d_kernel_size=4,
            #         bias_init="powerlaw_blockdependent",
            #     ),
            #     feedforward=FeedForwardConfig(
            #         proj_factor=1.3,
            #         act_fn="gelu",
            #     ),
            # ),
            context_length=context_length,
            num_blocks=depth,
            embedding_dim=dim,
            dropout=dropout_rate,
            slstm_at=[],
            add_post_blocks_norm=add_post_blocks_norm,
        ),
    )
