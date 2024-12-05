// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Error
// message `The implementation handling parallel_for with reduction requires
// work group size not bigger than 1` on Nvidia.

// XFAIL: hip_nvidia
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/14973 
// This test performs basic checks of parallel_for(nd_range, reduction, func)
// used with 'double' type.

#include "reduction_nd_ext_type.hpp"

int main() { return runTests<double>(sycl::aspect::fp64); }
