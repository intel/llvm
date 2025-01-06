// using --offload-compress without zstd should throw an error.
// REQUIRES: !zstd
// REQUIRES: build-and-run-mode
// RUN: not %{build} %O0 -g --offload-compress %S/Inputs/single_kernel.cpp -o %t_compress.out 2>&1 | FileCheck %s
// CHECK: '--offload-compress' option is specified but zstd is not available. The device image will not be compressed.
