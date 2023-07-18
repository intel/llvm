// https://github.com/intel/llvm/issues/10369
// UNSUPPORTED: gpu
//
// REQUIRES: gpu, linux

// UNSUPPORTED: ze_debug

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %S/Inputs/FindPrimesSYCL.cpp %S/Inputs/main.cpp -o %t.out -lpthread
// RUN: %{run} %t.out
