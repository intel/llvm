// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// CUDA and HIP backends have had no support for the generic address space yet
// XFAIL: cuda || hip

#include "and.h"

int main() { and_test_all<access::address_space::generic_space>(); }
