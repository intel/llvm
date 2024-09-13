// using --offload-compress without zstd should throw an error.
// REQUIRES: !zstd
// RUN: %{build} -O0 -g --offload-compress %S/Inputs/single_kernel.cpp -o %t_compress.out 2>&1 | FileCheck %s

// XFAIL: *
