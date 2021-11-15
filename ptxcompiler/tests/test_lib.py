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

import pytest
import sys

from ptxcompiler import _ptxcompilerlib


PTX_CODE = """\
.version 7.4
.target sm_52
.address_size 64

        // .globl       _Z1kPf

.visible .entry _Z1kPf(
        .param .u64 _Z1kPf_param_0
)
{
        .reg .b32       %r<2>;
        .reg .b64       %rd<3>;


        ld.param.u64    %rd1, [_Z1kPf_param_0];
        cvta.to.global.u64      %rd2, %rd1;
        mov.u32         %r1, 1065353216;
        st.global.u32   [%rd2], %r1;
        ret;

}
"""

OPTIONS = ('--gpu-name=sm_75',)


def test_get_version():
    major, minor = _ptxcompilerlib.get_version()
    # Any version less than 11.1 indicates an issue, since the library was
    # first available in 11.1
    assert (major, minor) >= (11, 1)


def test_create():
    handle = _ptxcompilerlib.create(PTX_CODE)
    assert handle != 0


def test_destroy():
    handle = _ptxcompilerlib.create(PTX_CODE)
    _ptxcompilerlib.destroy(handle)


def test_compile():
    # Check that compile does not error
    handle = _ptxcompilerlib.create(PTX_CODE)
    _ptxcompilerlib.compile(handle, OPTIONS)


def test_compile_options():
    options = ('--gpu-name=sm_75', '--device-debug')
    handle = _ptxcompilerlib.create(PTX_CODE)
    _ptxcompilerlib.compile(handle, options)
    compiled_program = _ptxcompilerlib.get_compiled_program(handle)
    assert b'nv_debug' in compiled_program


def test_compile_options_bad_option():
    options = ('--gpu-name=sm_75', '--bad-option')
    handle = _ptxcompilerlib.create(PTX_CODE)
    with pytest.raises(RuntimeError,
                       match="NVPTXCOMPILE_ERROR_COMPILATION_FAILURE error"):
        _ptxcompilerlib.compile(handle, options)
    error_log = _ptxcompilerlib.get_error_log(handle)
    assert "Unknown option" in error_log


def test_get_error_log():
    bad_ptx = ".target sm_52"
    handle = _ptxcompilerlib.create(bad_ptx)
    with pytest.raises(RuntimeError):
        _ptxcompilerlib.compile(handle, OPTIONS)

    error_log = _ptxcompilerlib.get_error_log(handle)
    assert "Missing .version directive" in error_log


def test_get_info_log():
    handle = _ptxcompilerlib.create(PTX_CODE)
    _ptxcompilerlib.compile(handle, OPTIONS)
    info_log = _ptxcompilerlib.get_info_log(handle)
    # Info log is empty
    assert "" == info_log


def test_get_compiled_program():
    handle = _ptxcompilerlib.create(PTX_CODE)
    _ptxcompilerlib.compile(handle, OPTIONS)
    compiled_program = _ptxcompilerlib.get_compiled_program(handle)
    # Check the compiled program is an ELF file by looking for the ELF header
    assert compiled_program[:4] == b'\x7fELF'


if __name__ == '__main__':
    sys.exit(pytest.main())
