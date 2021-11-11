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

from numba.cuda.cudadrv import devices
from ptxcompiler.api import compile_ptx
from numba.cuda import codegen


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


def patch_numba_codegen():
    codegen.JITCUDACodegen._library_class = PTXStaticCompileCodeLibrary
