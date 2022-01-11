// REQUIRES: gpu, linux
// UNSUPPORTED: cuda || hip
// UNSUPPORTED: ze_debug-1,ze_debug4
//
// RUN: %clangxx -fsycl %S/Inputs/FindPrimesSYCL.cpp %S/Inputs/main.cpp -o %t.out -lpthread
// RUN: %GPU_RUN_PLACEHOLDER %t.out
