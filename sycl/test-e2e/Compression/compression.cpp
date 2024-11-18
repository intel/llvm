// End-to-End test for testing device image compression.
// REQUIRES: zstd

// RUN: %{build} %O0 -g %S/Inputs/single_kernel.cpp -o %t_not_compress.out
// RUN: %{build} %O0 -g --offload-compress --offload-compression-level=3 %S/Inputs/single_kernel.cpp -o %t_compress.out
// RUN: %{run} %t_not_compress.out
// RUN: %{run} %t_compress.out
// RUN: not diff %t_not_compress.out %t_compress.out
