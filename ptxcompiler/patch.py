# Copyright (c) 2021, NVIDIA CORPORATION.
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
import subprocess
import sys

from ctypes import byref, c_int
from numba import config, cuda
from numba.cuda import codegen
from numba.cuda.cudadrv import devices
from ptxcompiler.api import compile_ptx

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
        lvl = str(config.CUDA_LOG_LEVEL).upper()
        lvl = getattr(logging, lvl, None)

        if not isinstance(lvl, int):
            # Default to critical level
            lvl = logging.CRITICAL
        logger.setLevel(lvl)

        # Did user specify a level?
        if config.CUDA_LOG_LEVEL:
            # Create a simple handler that prints to stderr
            handler = logging.StreamHandler(sys.stderr)
            fmt = '== CUDA [%(relativeCreated)d] %(levelname)5s -- %(message)s'
            handler.setFormatter(logging.Formatter(fmt=fmt))
            logger.addHandler(handler)
        else:
            # Otherwise, put a null handler
            logger.addHandler(logging.NullHandler())

    _logger = logger
    return logger


class PTXStaticCompileCodeLibrary(codegen.CUDACodeLibrary):
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

        arch = f'sm_{cc[0]}{cc[1]}'
        options = [f'--gpu-name={arch}']

        if self._max_registers:
            options.append(f'--maxrregcount={self._max_registers}')

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


def patch_needed():
    logger = get_logger()

    cp = subprocess.run([sys.executable, '-c', CMD], capture_output=True)

    if cp.returncode:
        msg = (f'Error getting driver and runtime versions:\n\nstdout:\n\n'
               f'{cp.stdout.decode()}\n\nstderr:\n\n{cp.stderr.decode()}\n\n'
               'Not patching Numba')
        logger.error(msg)
        return False

    versions = [int(s) for s in cp.stdout.strip().split()]
    driver_version = tuple(versions[:2])
    runtime_version = tuple(versions[2:])

    logger.debug("CUDA Driver version %s.%s" % driver_version)
    logger.debug("CUDA Runtime version %s.%s" % runtime_version)

    return driver_version < runtime_version


def patch_numba_codegen_if_needed():
    if patch_needed():
        logger = get_logger()
        debug_msg = "Patching Numba codegen for forward compatibility"
        logger.debug(debug_msg)
        codegen.JITCUDACodegen._library_class = PTXStaticCompileCodeLibrary
