# AGENTS.md

Guidance for AI coding agents working in this repository. This file focuses on
the SYCL/DPC++ parts of the tree; for anything else, the upstream LLVM
conventions apply.

## What this repository is

This is [intel/llvm](https://github.com/intel/llvm), Intel's LLVM-based
implementation of SYCL (DPC++). It is a fork of llvm/llvm-project with
SYCL-specific projects added on top (`sycl`, `libdevice`, `llvm-spirv`,
`sycl-jit`, `unified-runtime`, `xpti`/`xptifw`, `opencl`).

- The development branch is **`sycl`**, not `main`. Base your work on
  `origin/sycl` (the intel/llvm remote) and target PRs at `sycl`.
- Changes that are not SYCL-specific — generic LLVM/Clang improvements — should
  go to llvm.org upstream first, not here. See
  [CONTRIBUTING.md](CONTRIBUTING.md).
- Every product change must come with a test change (new, extended, or
  modified). This is enforced by review.

## Key documents (read before editing)

| Topic | Document |
| --- | --- |
| Build & run from source | [sycl/doc/GetStartedGuide.md](sycl/doc/GetStartedGuide.md) |
| DPC++ contribution rules | [sycl/doc/developer/ContributeToDPCPP.md](sycl/doc/developer/ContributeToDPCPP.md) |
| ABI/API stability rules | [sycl/doc/developer/ABIPolicyGuide.md](sycl/doc/developer/ABIPolicyGuide.md) |
| Extension lifecycle & spec authoring | [sycl/doc/extensions/README-process.md](sycl/doc/extensions/README-process.md), [sycl/doc/extensions/template.asciidoc](sycl/doc/extensions/template.asciidoc) |
| E2E test infrastructure | [sycl/test-e2e/README.md](sycl/test-e2e/README.md) |
| Runtime/compiler design notes | [sycl/doc/design/](sycl/doc/design/) |
| Env vars for debugging | [sycl/doc/EnvironmentVariables.md](sycl/doc/EnvironmentVariables.md) |
| Release notes | [sycl/ReleaseNotes.md](sycl/ReleaseNotes.md) |

## Layout of the SYCL parts

```
sycl/include/sycl/        Public SYCL headers (sycl/sycl.hpp is the entry point)
sycl/include/sycl/ext/    Extension headers: oneapi/, intel/, codeplay/
sycl/include/sycl/detail/ Implementation details, not user-facing API
sycl/source/              libsycl runtime sources
sycl/source/detail/       Runtime internals: scheduler, program manager,
                          adapter_impl (UR calls), memory management, ...
sycl/test/                Device-independent LIT tests (target: check-sycl)
sycl/unittests/           googletest unit tests for the runtime (check-sycl-unittests)
sycl/test-e2e/            End-to-end tests, need real devices/backends
sycl/doc/extensions/      Extension specifications (asciidoc)
clang/lib/Sema/SemaSYCL.cpp, clang/lib/Driver/  Front end / driver SYCL support
llvm/lib/SYCLLowerIR/     SYCL-specific LLVM IR passes
libdevice/                Device libraries (math, imf, sanitizers, ...)
unified-runtime/          Unified Runtime (UR): the L0/OpenCL/CUDA/HIP adapters
sycl-jit/                 Runtime JIT compilation / kernel fusion support
xpti/, xptifw/            Tracing framework
```
