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

import os
import os.path
import shutil
import versioneer
from distutils.sysconfig import get_config_var, get_python_inc
from setuptools import setup, Extension

include_dirs=[os.path.dirname(get_python_inc())]
library_dirs=[get_config_var("LIBDIR")]

# Find and add CUDA include paths
CUDA_HOME = os.environ.get("CUDA_HOME", False)
if not CUDA_HOME:
    path_to_cuda_gdb = shutil.which("cuda-gdb")
    if path_to_cuda_gdb is None:
        raise OSError(
            "Could not locate CUDA. "
            "Please set the environment variable "
            "CUDA_HOME to the path to the CUDA installation "
            "and try again."
        )
    CUDA_HOME = os.path.dirname(os.path.dirname(path_to_cuda_gdb))
if not os.path.isdir(CUDA_HOME):
    raise OSError(f"Invalid CUDA_HOME: directory does not exist: {CUDA_HOME}")
include_dirs.append(os.path.join(CUDA_HOME, "include"))
library_dirs.append(os.path.join(CUDA_HOME, "lib64"))

module = Extension(
    'ptxcompiler._ptxcompilerlib',
    sources=['ptxcompiler/_ptxcompilerlib.cpp'],
    include_dirs=include_dirs,
    libraries=['nvptxcompiler_static'],
    library_dirs=library_dirs,
    extra_compile_args=['-Wall', '-Werror'],
)

if "RAPIDS_PY_WHEEL_CUDA_SUFFIX" in os.environ:
    # borrow a similar hack from dask-cuda: https://github.com/rapidsai/dask-cuda/blob/b3ed9029a1ad02a61eb7fbd899a5a6826bb5cfac/setup.py#L12-L31
    orig_get_versions = versioneer.get_versions

    version_override = os.environ.get("RAPIDS_PY_WHEEL_VERSIONEER_OVERRIDE", "")

    def get_versions():
        data = orig_get_versions()
        if version_override != "":
            data["version"] = version_override
        return data

    versioneer.get_versions = get_versions

setup(
    name=f"ptxcompiler{os.getenv('RAPIDS_PY_WHEEL_CUDA_SUFFIX', default='')}",
    version=os.getenv("RAPIDS_PY_WHEEL_VERSIONEER_OVERRIDE", default=versioneer.get_version()),
    license="Apache 2.0",
    cmdclass=versioneer.get_cmdclass(),
    description='NVIDIA PTX Compiler binding',
    ext_modules=[module],
    packages=['ptxcompiler', 'ptxcompiler.tests'],
    extras_require={"test": ["pytest", "numba"]},
)
