# Copyright (c) 2021-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
import subprocess
import sys
import warnings

from ptxcompiler.api import compile_ptx

_numba_version_ok = False
_numba_error = None

min_numba_ver = (0, 54)
max_numba_ver = (0, 56)
NO_DRIVER = (math.inf, math.inf)

mvc_docs_url = ("https://numba.readthedocs.io/en/stable/cuda/"
                "minor_version_compatibility.html")

try:
    import numba

    ver = numba.version_info.short
    if ver < min_numba_ver:
        _numba_error = (
            f"version {numba.__version__} is insufficient for "
            "ptxcompiler patching - at least "
            "%s.%s is needed." % min_numba_ver
        )
    elif ver > max_numba_ver:
        _numba_error = (
            f"version {numba.__version__} should not be patched. "
            "Set the environment variable "
            "NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1 instead. "
            f"See {mvc_docs_url} for more details.")
    else:
        _numba_version_ok = True
except ImportError as ie:
    _numba_error = f"failed to import Numba: {ie}."

if _numba_version_ok:
    from numba.cuda import codegen
    from numba.cuda.codegen import CUDACodeLibrary
    from numba.cuda.cudadrv import devices
else:
    # Prevent the definition of PTXStaticCompileCodeLibrary failing if we have
    # no Numba CUDACodeLibrary - it won't be used anyway
    CUDACodeLibrary = object

_logger = None


# Create a logger that reports messages based on the value of Numba's config
# variable NUMBA_CUDA_LOG_LEVEL, so we can trace patching when tracing other
# CUDA operations.
def get_logger():
    global _logger

    if _logger:
        return _logger

    logger = logging.getLogger(__name__)

    # Create a default configuration if none exists already
    if not logger.hasHandlers():
        lvl = str(numba.config.CUDA_LOG_LEVEL).upper()
        lvl = getattr(logging, lvl, None)

        if not isinstance(lvl, int):
            # Default to critical level
            lvl = logging.CRITICAL
        logger.setLevel(lvl)

        # Did user specify a level?
        if numba.config.CUDA_LOG_LEVEL:
            # Create a simple handler that prints to stderr
            handler = logging.StreamHandler(sys.stderr)
            fmt = (
                "== CUDA (ptxcompiler) [%(relativeCreated)d] "
                "%(levelname)5s -- %(message)s"
            )
            handler.setFormatter(logging.Formatter(fmt=fmt))
            logger.addHandler(handler)
        else:
            # Otherwise, put a null handler
            logger.addHandler(logging.NullHandler())

    _logger = logger
    return logger


class PTXStaticCompileCodeLibrary(CUDACodeLibrary):
    def get_cubin(self, cc=None):
        if cc is None:
            ctx = devices.get_context()
            device = ctx.device
            cc = device.compute_capability

        cubin = self._cubin_cache.get(cc, None)
        if cubin:
            return cubin

        ptxes = self._get_ptxes(cc=cc)
        if len(ptxes) > 1:
            msg = "Cannot link multiple PTX files with forward compatibility"
            raise RuntimeError(msg)

        arch = f"sm_{cc[0]}{cc[1]}"
        options = [f"--gpu-name={arch}"]

        if self._max_registers:
            options.append(f"--maxrregcount={self._max_registers}")

        # Compile PTX to cubin
        ptx = ptxes[0]
        res = compile_ptx(ptx, options)
        cubin = res.compiled_program

        return cubin


CMD = """\
from ctypes import c_int, byref
from numba import cuda
dv = c_int(0)
cuda.cudadrv.driver.driver.cuDriverGetVersion(byref(dv))
drv_major = dv.value // 1000
drv_minor = (dv.value - (drv_major * 1000)) // 10
run_major, run_minor = cuda.runtime.get_version()
print(f'{drv_major} {drv_minor} {run_major} {run_minor}')
"""



def patch_forced_by_user():
    logger = get_logger()
    # The patch is needed if the user explicitly forced it with an environment
    # variable.
    apply = os.getenv("PTXCOMPILER_APPLY_NUMBA_CODEGEN_PATCH")
    if apply is not None:
        logger.debug(f"PTXCOMPILER_APPLY_NUMBA_CODEGEN_PATCH={apply}")
        try:
            apply = int(apply)
        except ValueError:
            apply = False

    return bool(apply)


def check_disabled_in_env():
    logger = get_logger()
    # We should avoid checking whether the patch is needed if the user
    # requested that we don't check (e.g. in a non-fork-safe environment)
    check = os.getenv("PTXCOMPILER_CHECK_NUMBA_CODEGEN_PATCH_NEEDED")
    if check is not None:
        logger.debug(f"PTXCOMPILER_CHECK_NUMBA_CODEGEN_PATCH_NEEDED={check}")
        try:
            check = int(check)
        except ValueError:
            check = False
    else:
        check = True

    return not check


def patch_needed():
    # If Numba is not present, we don't need the patch.
    # We also can't use it to check driver and runtime versions, so exit early.
    if not _numba_version_ok:
        return False
    if patch_forced_by_user():
        return True
    if check_disabled_in_env():
        return False
    else:
        # Check whether the patch is needed by comparing the driver and runtime
        # versions - it is needed if the runtime version exceeds the driver
        # version.
        driver_version, runtime_version = get_versions()
        return driver_version < runtime_version


def get_versions():
    logger = get_logger()
    cp = subprocess.run([sys.executable, "-c", CMD], capture_output=True)
    if cp.returncode:
        msg = (
            f"Error getting driver and runtime versions:\n\nstdout:\n\n"
            f"{cp.stdout.decode()}\n\nstderr:\n\n{cp.stderr.decode()}\n\n"
            "Not patching Numba"
        )
        logger.error(msg)
        return NO_DRIVER

    versions = [int(s) for s in cp.stdout.strip().split()]
    driver_version = tuple(versions[:2])
    runtime_version = tuple(versions[2:])

    logger.debug("CUDA Driver version %s.%s" % driver_version)
    logger.debug("CUDA Runtime version %s.%s" % runtime_version)

    return driver_version, runtime_version


def safe_get_versions():
    """
    Return a 2-tuple of deduced driver and runtime versions.

    To ensure that this function does not initialize a CUDA context, calls to the
    runtime and driver are made in a subprocess.

    If PTXCOMPILER_CHECK_NUMBA_CODEGEN_PATCH_NEEDED is set
    in the environment, then this subprocess call is not launched. To specify the
    driver and runtime versions of the environment in this case, set
    PTXCOMPILER_KNOWN_DRIVER_VERSION and PTXCOMPILER_KNOWN_RUNTIME_VERSION
    appropriately.
    """
    if check_disabled_in_env():
        try:
            # allow user to specify driver/runtime versions manually, if necessary
            driver_version = os.environ["PTXCOMPILER_KNOWN_DRIVER_VERSION"].split(".")
            runtime_version = os.environ["PTXCOMPILER_KNOWN_RUNTIME_VERSION"].split(".")
            driver_version, runtime_version = (
                tuple(map(int, driver_version)),
                tuple(map(int, runtime_version))
            )
        except (KeyError, ValueError):
            warnings.warn(
                "No way to determine driver and runtime versions for patching, "
                "set PTXCOMPILER_KNOWN_DRIVER_VERSION/PTXCOMPILER_KNOWN_RUNTIME_VERSION"
            )
            return NO_DRIVER
    else:
        driver_version, runtime_version = get_versions()
    return driver_version, runtime_version


def patch_numba_codegen_if_needed():
    if not _numba_version_ok:
        msg = f"Cannot patch Numba: {_numba_error}"
        raise RuntimeError(msg)

    logger = get_logger()

    if patch_needed():
        logger.debug("Patching Numba codegen for forward compatibility")
        codegen.JITCUDACodegen._library_class = PTXStaticCompileCodeLibrary
    else:
        logger.debug("Not patching Numba codegen")
