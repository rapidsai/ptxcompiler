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
import versioneer
from distutils.sysconfig import get_config_var, get_python_inc
from setuptools import setup, Extension

module = Extension(
    'ptxcompiler._ptxcompilerlib',
    sources=['ptxcompiler/_ptxcompilerlib.cpp'],
    include_dirs=[os.path.dirname(get_python_inc())],
    libraries=['nvptxcompiler_static'],
    library_dirs=[get_config_var("LIBDIR")],
    extra_compile_args=['-Wall', '-Werror'],
)

setup(
    name='ptxcompiler',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='NVIDIA PTX Compiler binding',
    ext_modules=[module],
    packages=['ptxcompiler', 'ptxcompiler.tests'],
)
