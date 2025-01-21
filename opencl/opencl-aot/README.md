# OpenCL ahead-of-time compilation tool (opencl-aot)

OpenCL ahead-of-time compilation tool (`opencl-aot`) is a tool which generates
device-dependent OpenCL program binary from SPIR-V binary primarily for Intel(R)
processor devices.

## Prerequisites

To use `opencl-aot` tool, you must have:

* OpenCL runtime installed for the target device (see
  [Install low level runtime](../sycl/doc/GetStartedGuide.md#install-low-level-runtime))
* OpenCL ICD Loader installed with support of OpenCL 2.1 or higher

## How to use

`opencl-aot` tool is integrated with Clang. To use ahead-of-time compilation for
SYCL application for Intel(R) processor device, run Clang with
`-fsycl-targets=spir64_x86_64` option.

To enable optimizations for target CPU architecture, add
`-Xsycl-target-backend "-march=<arch>"` option, where `<arch>` could be
instruction set, e.g., `avx512`, `avx2`, `avx` or `sse4.2` and available CPUs,
e.g., `wsm`, `snb`, `bdw`, `adl` or `skx`. Execute `opencl-aot --help` to get
full lists.

To execute `opencl-aot` directly for Intel(R) processor device, run it with
`<spv> -o=<output> --device=cpu`, where `<spv>` is path to SPIR-V binary file,
`<output>` is path to created OpenCL program binary file.

To generate SPIR-V binary file from OpenCL device kernel code, use Clang to
generate LLVM IR and pass it to
[llvm-spirv](https://github.com/KhronosGroup/SPIRV-LLVM-Translator) tool. For
more information about generation LLVM IR from OpenCL device kernel code, see
[OpenCL Features](https://clang.llvm.org/docs/UsersManual.html#opencl-features)
of Clang Compiler User's Manual.

For more information about `opencl-aot` tool, execute it with `--help` option.
