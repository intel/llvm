// REQUIRES: aspect-fp16
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Error message on Nvidia:
// `The implementation handling parallel_for with reduction requires
// work group size not bigger than 1`.
// XFAIL: hip_nvidia
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/14973

// Windows doesn't yet have full shutdown().
// UNSUPPORTED: ze_debug && windows

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// used with 'half' type.

#include "reduction_nd_ext_type.hpp"

int main() { return runTests<sycl::half>(sycl::aspect::fp16); }
