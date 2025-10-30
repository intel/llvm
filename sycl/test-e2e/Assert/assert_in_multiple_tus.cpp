// REQUIRES: linux

// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/7634
//
// UNSUPPORTED: cuda
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/8832
//
// XFAIL: (opencl && gpu)
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/11364
//
// L0 does not currently abort after synchronizing with a failing kernel.
// UNSUPPORTED: level_zero
// UNSUPPORTED-TRACKER: GSD-11097

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

// RUN: %{build} -I %S/Inputs %S/Inputs/kernels_in_file2.cpp -o %t.out
// RUN: %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt
//
// CUDA uses block/thread vs global/local id for SYCL, also it shows the
// position of a thread within the block, not the absolute ID.
// CHECK:      {{.*}}kernels_in_file2.cpp:15: int calculus(int): {{global id: \[5|block: \[1}},0,0], {{local id|thread}}: [1,0,0]
// CHECK-SAME: Assertion `X && "this message from calculus"` failed
// CHECK-NOT:  this message from file2
// CHECK-NOT:  The test ended.

#include "assert_in_multiple_tus.hpp"
