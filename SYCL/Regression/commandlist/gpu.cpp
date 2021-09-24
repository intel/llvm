// REQUIRES: gpu, linux
// UNSUPPORTED: cuda || hip
//
// RUN: %clangxx -fsycl %S/Inputs/FindPrimesSYCL.cpp %S/Inputs/main.cpp -o %t.out -lpthread
// RUN: %GPU_RUN_PLACEHOLDER %t.out
