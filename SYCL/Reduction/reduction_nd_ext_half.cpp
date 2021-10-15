// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// Missing __spirv_GroupFAdd, __spirv_GroupFMin, __spirv_GroupFMax on AMD, error
// message `The implementation handling parallel_for with reduction requires
// work group size not bigger than 1` on Nvidia.
// XFAIL: hip_amd || hip_nvidia

// TODO: Enable the test for HOST when it supports intel::reduce() and barrier()
// RUNx: %HOST_RUN_PLACEHOLDER %t.out

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// used with 'half' type.

#include "reduction_nd_ext_type.hpp"

int main() { return runTests<sycl::half>(sycl::aspect::fp16); }
