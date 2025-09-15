> [!IMPORTANT]
> The final release of this project was `v0.8.1`.
> The core functionality of this project is replaced by the nvjitlink library in CUDA 12+.
> Bindings for nvjitlink are available in `cuda.core`, and for
> `numba-cuda>=0.16`, `numba-cuda` automatically detects and enables nvjitlink
> when needed and available with no explicit configuration.
> See https://github.com/rapidsai/ptxcompiler/issues/42 for details.

# Static PTX Compiler Python binding and Numba patch

This package provides a Python binding for the `libptxcompiler_static.a` library
and a Numba patch that fixes it to use the static library for compiling PTX
instead of the linker. This enables Numba to support CUDA enhanced
compatibility for scenarios where a single PTX file is compiled and linked as
part of the compilation process. This covers all use cases, except:

- Using Cooperative Groups.
- Debugging features - this includes debug and lineinfo generation, and
  exception checking inside CUDA kernels.


## Numba support

Numba 0.54.1 and above are supported.


## Installation

Install with either:

```
python setup.py develop
```

or

```
python setup.py install
```


## Testing

Run

```
pytest
```

or

```
python ptxcompiler/tests/test_lib.py
```


## Usage

### Numba >= 0.57

To configure Numba to use ptxcompiler, set the environment variable
`NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1`. See the Numba [CUDA Minor
Version Compatibility
documentation](https://numba.readthedocs.io/en/stable/cuda/minor_version_compatibility.html)
for further information.

### Numba versions < 0.57

Numba versions < 0.57 need to be monkey patched to use ptxcompiler if required.
To apply the monkey patch if needed, call the
`patch_numba_codegen_if_needed()` function:

```python
from ptxcompiler.patch import patch_numba_codegen_if_needed
patch_numba_codegen_if_needed()
```

This function spawns a new process to check the CUDA Driver and Runtime
versions, so it can be safely called at any point in a process. It will only
patch Numba when the Runtime version exceeds the Driver version.

Under certain circumstances (for example running with InfiniBand
network stacks), spawning a subprocess might not be possible. For
these cases, the patching behaviour can be controlled using two
environment variables:

- `PTXCOMPILER_CHECK_NUMBA_CODEGEN_PATCH_NEEDED`: if set to a truthy
  integer then a subprocess will be spawned to check if patching Numba
  is necessary. Default value: True (the subprocess check is carried out)
- `PTXCOMPILER_APPLY_NUMBA_CODEGEN_PATCH`: if it is known that
  patching is necessary, but spawning a subprocess is not possible,
  set this to a truthy integer to unconditionally patch Numba. Default
  value: False (Numba is not unconditionally patched).
