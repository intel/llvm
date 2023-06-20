// REQUIRES: gpu, linux
// UNSUPPORTED: hip
// UNSUPPORTED: ze_debug
//
// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %S/Inputs/FindPrimesSYCL.cpp %S/Inputs/main.cpp -o %t.out -lpthread
// RUN: %{run} %t.out
