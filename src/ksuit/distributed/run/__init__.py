from .managed import run_managed
from .unmanaged import run_unmanaged


def run(main, devices=None, accelerator="gpu", master_port=None, mig_devices=None):
    from ksuit.distributed import is_managed
    if is_managed():
        run_managed(
            main=main,
            accelerator=accelerator,
            devices=devices,
        )
    else:
        run_unmanaged(
            main=main,
            accelerator=accelerator,
            devices=devices,
            master_port=master_port,
            mig_devices=mig_devices,
        )
