// Test to check warnings when using --offload-compress without zstd.
// REQUIRES: !zstd
// RUN: %{build} -O0 -g --offload-compress %S/Inputs/single_kernel.cpp -o %t_compress.out 2>&1 | FileCheck %s

// CHECK: warning: '--offload-compress' option is specified but zstd is not available. The device image will not be compressed.
