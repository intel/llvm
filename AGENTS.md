# AGENTS.md

Guidance for AI coding agents working in this repository. This file focuses on
the SYCL/DPC++ parts of the tree; for anything else, the upstream LLVM
conventions apply.

## What this repository is

This is [intel/llvm](https://github.com/intel/llvm), Intel's LLVM-based
implementation of SYCL (DPC++). It is a fork of llvm/llvm-project with
SYCL-specific projects added on top (`sycl`, `libdevice`, `llvm-spirv`,
`sycl-jit`, `unified-runtime`, `xpti`/`xptifw`, `opencl`).

- The development branch is **`sycl`**, not `main`. 
- Follow sycl/doc/developer/ContributeToDPCPP.md for your contributions.

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
