// End-to-End test for testing device image compression in AOT.
// REQUIRES: zstd, opencl-aot, cpu

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 %O0 --offload-compress --offload-compression-level=3 %S/Inputs/single_kernel.cpp -o %t_compress.out
// RUN: %{run} %t_compress.out
