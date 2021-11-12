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
import multiprocessing as mp
import sys
import traceback

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


def get_driver_and_runtime_versions():
    # Numba doesn't provide a convenient function to get the driver version, so
    # we need to access it through its ctypes binding
    dv = c_int(0)
    cuda.cudadrv.driver.driver.cuDriverGetVersion(byref(dv))
    major = dv.value // 1000
    minor = (dv.value - (major * 1000)) // 10
    driver_version = (major, minor)
    runtime_version = cuda.runtime.get_version()
    return (driver_version, runtime_version)


def get_versions_wrapper(result_queue):
    try:
        output = get_driver_and_runtime_versions()
        success = True
    # We catch all exceptions so that they can be propagated
    except:  # noqa: E722
        # Record the exception as a string for the caller to report to the user
        output = traceback.format_exc()
        success = False

    result_queue.put((success, output))


def patch_needed():
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    proc = ctx.Process(target=get_versions_wrapper, args=(result_queue,))
    proc.start()
    proc.join()
    success, output = result_queue.get()

    if not success:
        msg = f"Error getting driver and runtime versions:\n{output}"
        raise RuntimeError(msg)

    driver_version, runtime_version = output

    logger = get_logger()
    logger.debug("CUDA Driver version %s.%s" % driver_version)
    logger.debug("CUDA Runtime version %s.%s" % runtime_version)

    return driver_version < runtime_version


def patch_numba_codegen_if_needed():
    if patch_needed():
        logger = get_logger()
        debug_msg = "Patching Numba codegen for forward compatibility"
        logger.debug(debug_msg)
        codegen.JITCUDACodegen._library_class = PTXStaticCompileCodeLibrary
