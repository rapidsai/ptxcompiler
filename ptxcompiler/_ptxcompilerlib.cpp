/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <new>
#include <nvPTXCompiler.h>
#include <string.h>

static PyObject *get_version(PyObject *self) {
  unsigned int major, minor;
  PyObject *py_major = nullptr, *py_minor = nullptr, *py_version = nullptr;

  nvPTXCompileResult res = nvPTXCompilerGetVersion(&major, &minor);
  if (res != NVPTXCOMPILE_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Error calling nvPTXCompilerGetVersion");
    return nullptr;
  }

  if ((py_major = PyLong_FromUnsignedLong(major)) == nullptr)
    goto error;

  if ((py_minor = PyLong_FromUnsignedLong(minor)) == nullptr)
    goto error;

  if ((py_version = PyTuple_Pack(2, py_major, py_minor)) == nullptr)
    goto error;

  return py_version;

error:
  Py_XDECREF(py_major);
  Py_XDECREF(py_minor);
  Py_XDECREF(py_version);
  return nullptr;
}

static PyObject *create(PyObject *self, PyObject *args) {
  PyObject *ret = nullptr;
  char *ptx_code;
  nvPTXCompilerHandle *compiler;

  if (!PyArg_ParseTuple(args, "s", &ptx_code))
    return nullptr;

  try {
    compiler = new nvPTXCompilerHandle;
  } catch (const std::bad_alloc &) {
    PyErr_NoMemory();
    return nullptr;
  }

  nvPTXCompileResult res =
      nvPTXCompilerCreate(compiler, strlen(ptx_code), ptx_code);
  if (res != NVPTXCOMPILE_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "Error calling nvPTXCompilerCreate");
    goto error;
  }

  if ((ret = PyLong_FromUnsignedLongLong((unsigned long long)compiler)) ==
      nullptr) {
    // Attempt to destroy the compiler - since we're already in an error
    // condition, there's no point in checking the return code and taking any
    // further action based on it though.
    nvPTXCompilerDestroy(compiler);
    goto error;
  }

  return ret;

error:
  delete compiler;
  return nullptr;
}

static PyObject *destroy(PyObject *self, PyObject *args) {
  nvPTXCompilerHandle *compiler;
  if (!PyArg_ParseTuple(args, "K", &compiler))
    return nullptr;

  nvPTXCompileResult res = nvPTXCompilerDestroy(compiler);

  if (res != NVPTXCOMPILE_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "Error calling nvPTXCompilerDestroy");
    return nullptr;
  }

  delete compiler;

  Py_RETURN_NONE;
}

static PyObject *compile(PyObject *self, PyObject *args) {
  nvPTXCompilerHandle *compiler;
  PyObject *options;
  if (!PyArg_ParseTuple(args, "KO!", &compiler, &PyTuple_Type, &options))
    return nullptr;

  Py_ssize_t n_options = PyTuple_Size(options);
  const char **compile_options = new char const *[n_options];
  for (Py_ssize_t i = 0; i < n_options; i++) {
    PyObject *item = PyTuple_GetItem(options, i);
    compile_options[i] = PyUnicode_AsUTF8AndSize(item, nullptr);
  }

  nvPTXCompileResult res =
      nvPTXCompilerCompile(*compiler, n_options, compile_options);

  delete[] compile_options;

  if (res != NVPTXCOMPILE_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "Error calling nvPTXCompilerCompile");
    return nullptr;
  }

  Py_RETURN_NONE;
}

static PyObject *get_error_log(PyObject *self, PyObject *args) {
  nvPTXCompilerHandle *compiler;
  if (!PyArg_ParseTuple(args, "K", &compiler))
    return nullptr;

  size_t error_log_size;
  nvPTXCompileResult res =
      nvPTXCompilerGetErrorLogSize(*compiler, &error_log_size);
  if (res != NVPTXCOMPILE_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Error calling nvPTXCompilerGetErrorLogSize");
    return nullptr;
  }

  // The size returned doesn't include a trailing null byte
  char *error_log = new char[error_log_size + 1];
  res = nvPTXCompilerGetErrorLog(*compiler, error_log);
  if (res != NVPTXCOMPILE_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Error calling nvPTXCompilerGetErrorLog");
    return nullptr;
  }

  PyObject *py_log = PyUnicode_FromStringAndSize(error_log, error_log_size);
  // Once we've copied the log to a Python object we can delete it - we don't
  // need to check whether creation of the Unicode object succeeded, because we
  // delete the log either way.
  delete[] error_log;

  return py_log;
}

static PyObject *get_info_log(PyObject *self, PyObject *args) {
  nvPTXCompilerHandle *compiler;
  if (!PyArg_ParseTuple(args, "K", &compiler))
    return nullptr;

  size_t info_log_size;
  nvPTXCompileResult res =
      nvPTXCompilerGetInfoLogSize(*compiler, &info_log_size);
  if (res != NVPTXCOMPILE_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Error calling nvPTXCompilerGetInfoLogSize");
    return nullptr;
  }

  // The size returned doesn't include a trailing null byte
  char *info_log = new char[info_log_size + 1];
  res = nvPTXCompilerGetInfoLog(*compiler, info_log);
  if (res != NVPTXCOMPILE_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Error calling nvPTXCompilerGetInfoLog");
    return nullptr;
  }

  PyObject *py_log = PyUnicode_FromStringAndSize(info_log, info_log_size);
  // Once we've copied the log to a Python object we can delete it - we don't
  // need to check whether creation of the Unicode object succeeded, because we
  // delete the log either way.
  delete[] info_log;

  return py_log;
}

static PyObject *get_compiled_program(PyObject *self, PyObject *args) {
  nvPTXCompilerHandle *compiler;
  if (!PyArg_ParseTuple(args, "K", &compiler))
    return nullptr;

  size_t compiled_program_size;
  nvPTXCompileResult res =
      nvPTXCompilerGetCompiledProgramSize(*compiler, &compiled_program_size);
  if (res != NVPTXCOMPILE_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Error calling nvPTXCompilerGetCompiledProgramSize");
    return nullptr;
  }

  char *compiled_program = new char[compiled_program_size];
  res = nvPTXCompilerGetCompiledProgram(*compiler, compiled_program);
  if (res != NVPTXCOMPILE_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Error calling nvPTXCompilerGetCompiledProgram");
    return nullptr;
  }

  PyObject *py_prog =
      PyBytes_FromStringAndSize(compiled_program, compiled_program_size);
  // Once we've copied the compiled program to a Python object we can delete it
  // - we don't need to check whether creation of the Unicode object succeeded,
  // because we delete the compiled program either way.
  delete[] compiled_program;

  return py_prog;
}

static PyMethodDef ext_methods[] = {
    {"get_version", (PyCFunction)get_version, METH_NOARGS,
     "Returns a tuple giving the version"},
    {"create", (PyCFunction)create, METH_VARARGS,
     "Returns a handle to a new compiler object"},
    {"destroy", (PyCFunction)destroy, METH_VARARGS,
     "Given a handle, destroy a compiler object"},
    {"compile", (PyCFunction)compile, METH_VARARGS,
     "Given a handle, compile the PTX"},
    {"get_error_log", (PyCFunction)get_error_log, METH_VARARGS,
     "Given a handle, return the error log"},
    {"get_info_log", (PyCFunction)get_info_log, METH_VARARGS,
     "Given a handle, return the info log"},
    {"get_compiled_program", (PyCFunction)get_compiled_program, METH_VARARGS,
     "Given a handle, return the compiled program"},
    {nullptr}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "ptxcompiler",
    "Provides access to PTX compiler API methods", -1, ext_methods};

PyMODINIT_FUNC PyInit__ptxcompilerlib(void) {
  PyObject *m = PyModule_Create(&moduledef);
  return m;
}
