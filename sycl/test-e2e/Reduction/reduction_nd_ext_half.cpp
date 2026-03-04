// REQUIRES: aspect-fp16
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// used with 'half' type.

#include "reduction_nd_ext_type.hpp"

int main() { return runTests<sycl::half>(sycl::aspect::fp16); }
