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

from ptxcompiler import _ptxcompilerlib
from collections import namedtuple


PTXCompilerResult = namedtuple(
    'PTXCompilerResult',
    ('compiled_program', 'info_log')
)


def compile_ptx(ptx, options):
    options = tuple(options)
    handle = _ptxcompilerlib.create(ptx)
    try:
        _ptxcompilerlib.compile(handle, options)
    except RuntimeError:
        error_log = _ptxcompilerlib.get_error_log(handle)
        _ptxcompilerlib.destroy(handle)
        raise RuntimeError(error_log)

    try:
        compiled_program = _ptxcompilerlib.get_compiled_program(handle)
        info_log = _ptxcompilerlib.get_info_log(handle)
    finally:
        _ptxcompilerlib.destroy(handle)

    return PTXCompilerResult(compiled_program=compiled_program,
                             info_log=info_log)
