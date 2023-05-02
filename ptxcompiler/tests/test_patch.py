# Copyright (c) 2022, NVIDIA CORPORATION.
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

import os
import pytest

from ptxcompiler import patch
from ptxcompiler.patch import (_numba_version_ok,
                               patch_numba_codegen_if_needed,
                               min_numba_ver, PTXStaticCompileCodeLibrary)
from unittest.mock import patch as mock_patch


def test_numba_patching_numba_not_ok():
    with mock_patch.multiple(
            patch,
            _numba_version_ok=False,
            _numba_error='<error>'):
        with pytest.raises(RuntimeError, match='Cannot patch Numba: <error>'):
            patch_numba_codegen_if_needed()


@pytest.mark.skipif(
    not _numba_version_ok,
    reason=f"Requires Numba >= {min_numba_ver[0]}.{min_numba_ver[1]}"
)
def test_numba_patching():
    # We import the codegen here rather than at the top level because the
    # import may fail if if Numba is not present or an unsupported version.
    from numba.cuda.codegen import JITCUDACodegen

    # Force application of the patch so we can test application regardless of
    # whether it is needed.
    os.environ['PTXCOMPILER_APPLY_NUMBA_CODEGEN_PATCH'] = '1'

    patch_numba_codegen_if_needed()
    assert JITCUDACodegen._library_class is PTXStaticCompileCodeLibrary
