# SYCL-JIT library

A library to make LLVM-based tooling available to the SYCL runtime.

## Current use-cases

- Materialization of specialization constants for third-party GPUs
  - Entrypoint declared in `jit-compiler/include/Materializer.h`
- Runtime-compilation of SYCL code (via `kernel_compiler` extension)
  - Entrypoints declared in `jit-compiler/include/RTC.h`

## Provided infrastructure

- CMake setup to link LLVM and Clang libraries and produce a pass plugin library
  - NB: Clang can be invoked programmatically via the LibTooling API
- Translation from LLVM module to SPIR-V and PTX/AMDGCN; the resulting
  "binaries" (i.e. SPIR-V blob or PTX/AMDGCN text) are managed by the
  `JITContext` singleton class
- Option handling
