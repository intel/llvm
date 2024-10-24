// REQUIRES: linux

// https://github.com/intel/llvm/issues/7634
// UNSUPPORTED: hip
//
// https://github.com/intel/llvm/issues/8832
// UNSUPPORTED: cuda
//
// XFAIL: (opencl && gpu)
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/11364

// https://github.com/intel/llvm/issues/15029
// UNSUPPORTED: gpu-intel-dg2

// RUN: %{build} -DSYCL_FALLBACK_ASSERT=1 -I %S/Inputs %S/Inputs/kernels_in_file2.cpp -o %t.out
// RUN: %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt %if fpga %{ --check-prefix=CHECK-ACC %}
// Shouldn't fail on ACC as fallback assert isn't enqueued there
//
// CUDA uses block/thread vs global/local id for SYCL, also it shows the
// position of a thread within the block, not the absolute ID.
// CHECK:      {{.*}}kernels_in_file2.cpp:15: int calculus(int): {{global id: \[5|block: \[1}},0,0], {{local id|thread}}: [1,0,0]
// CHECK-SAME: Assertion `X && "this message from calculus"` failed
// CHECK-NOT:  this message from file2
// CHECK-NOT:  The test ended.
//
// CHECK-ACC-NOT: {{.*}}kernels_in_file2.cpp:15: int calculus(int): global id: [5,0,0], local id: [1,0,0]
// CHECK-ACC: The test ended.

#include "assert_in_multiple_tus.hpp"
