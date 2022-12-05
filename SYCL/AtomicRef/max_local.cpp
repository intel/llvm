// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// HIP does not support floating point atomics.
// UNSUPPORTED: hip

#include "max.h"

int main() { max_test_all<access::address_space::local_space>(); }
