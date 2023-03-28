// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// Error
// message `The implementation handling parallel_for with reduction requires
// work group size not bigger than 1` on Nvidia.

// XFAIL: hip_nvidia

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// used with 'double' type.

#include "reduction_nd_ext_type.hpp"

int main() { return runTests<double>(sycl::aspect::fp64); }
