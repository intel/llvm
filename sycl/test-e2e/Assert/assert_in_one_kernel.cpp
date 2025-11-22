// REQUIRES: linux

// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/7634

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt
//
// CHECK:      {{.*}}assert_in_one_kernel.hpp:12: void kernelFunc(int *, int): {{.*}} [{{[0-3]}},0,0], {{.*}} [0,0,0]
// CHECK-SAME: Assertion `Buf[wiID] != 0 && "from assert statement"` failed
// CHECK-NOT:  The test ended.

#include "assert_in_one_kernel.hpp"
