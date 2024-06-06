import logging
import sys

import packaging.version

expected_torch = "2.0.0"
expected_torchvision = "0.15.0"
expected_torchmetrics_version = "0.11.0"
expected_python_major = 3
expected_python_minor = 9
expected_kappaschedules = "0.0.31"
expected_kappamodules = "0.1.70"


def check_versions(verbose):
    log_fn = logging.info if verbose else lambda _: None

    log_fn("------------------")
    log_fn("VERSION CHECK")
    # print python environment path
    executable_log_fn = logging.info if verbose else print
    executable_log_fn(f"executable: {sys.executable}")

    # python version >= 3.7 for order preserving dict (https://docs.python.org/3/whatsnew/3.7.html)
    py_version = sys.version_info
    assert py_version.major >= expected_python_major and py_version.minor >= expected_python_minor, \
        f"upgrade python ({py_version.major}.{py_version.minor} < {expected_python_major}.{expected_python_minor})"
    log_fn(f"python version: {py_version.major}.{py_version.minor}.{py_version.micro}")

    #
    import torch
    log_fn(f"torch version: {torch.__version__}")
    assert packaging.version.parse(torch.__version__) >= packaging.version.parse(expected_torch)
    if verbose and torch.cuda.is_available():
        log_fn(f"torch.cuda version: {torch.version.cuda}")
    import torchvision
    assert packaging.version.parse(torchvision.__version__) >= packaging.version.parse(expected_torchvision)
    log_fn(f"torchvision.version: {torchvision.__version__}")

    # pip
    def _check_pip_dependency(actual_version, expected_version, pip_dependency_name):
        assert packaging.version.parse(actual_version) >= packaging.version.parse(expected_version), (
            f"upgrade {pip_dependency_name} with 'pip install {pip_dependency_name} --upgrade' "
            f"({actual_version} < {expected_version})"
        )
        log_fn(f"{pip_dependency_name} version: {actual_version}")

    import torchmetrics
    _check_pip_dependency(torchmetrics.__version__, expected_torchmetrics_version, "torchmetrics")
    import kappaschedules
    _check_pip_dependency(kappaschedules.__version__, expected_kappaschedules, "kappaschedules")
    import kappamodules
    _check_pip_dependency(kappamodules.__version__, expected_kappamodules, "kappamodules")
