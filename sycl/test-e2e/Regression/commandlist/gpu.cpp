// REQUIRES: gpu, linux
// UNSUPPORTED: hip
// UNSUPPORTED: ze_debug-1,ze_debug4
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %S/Inputs/FindPrimesSYCL.cpp %S/Inputs/main.cpp -o %t.out -lpthread
// RUN: %GPU_RUN_PLACEHOLDER %t.out
