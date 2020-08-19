// REQUIRES: L0_plugin
// REQUIRES: gpu
//
// RUN: %clangxx -fsycl %S/Inputs/FindPrimesSYCL.cpp %S/Inputs/main.cpp -o %t.out -lpthread -lOpenCL
// RUN: %GPU_RUN_PLACEHOLDER %t.out
