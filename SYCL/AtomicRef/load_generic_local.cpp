// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t.out -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_70
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// CUDA backend has had no support for the generic address space yet. Barrier is
// not supported on host.
// XFAIL: cuda, hip, host

#define TEST_GENERIC_IN_LOCAL 1

#include "load.h"

int main() { load_test_all<access::address_space::generic_space>(); }
