// using --offload-compress without zstd should throw an error.
// REQUIRES: !zstd
// RUN: not %{build} %O0 -g --offload-compress %S/Inputs/single_kernel.cpp -o %t_compress.out 2>&1 | FileCheck %s
// CHECK: error: '--offload-compress' is specified but the compiler is built without zstd support.
// CHECK-NEXT: If you are using a custom DPC++ build, please refer to https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain-with-device-image-compression-support for more information on how to build with zstd support.
