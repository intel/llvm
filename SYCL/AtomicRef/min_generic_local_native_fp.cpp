// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// HIP backend has had no support for the generic address space yet.
// HIP does not support native floating point atomics
// XFAIL: hip

#define SYCL_USE_NATIVE_FP_ATOMICS
#define FP_TESTS_ONLY
#define TEST_GENERIC_IN_LOCAL 1

#include "min.h"

int main() { min_test_all<access::address_space::generic_space>(); }
