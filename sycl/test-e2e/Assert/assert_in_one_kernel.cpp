// REQUIRES: linux

// https://github.com/intel/llvm/issues/7634
// UNSUPPORTED: hip
//
// FIXME: Remove XFAIL one intel/llvm#11364 is resolved
// XFAIL: (opencl && gpu)

// RUN: %{build} -DSYCL_FALLBACK_ASSERT=1 -o %t.out
// Shouldn't fail on ACC as fallback assert isn't enqueued there
// RUN: %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt %if fpga %{ --check-prefix=CHECK-ACC %}
//
// CHECK:      {{.*}}assert_in_one_kernel.hpp:10: void kernelFunc(int *, int): {{.*}} [{{[0-3]}},0,0], {{.*}} [0,0,0]
// CHECK-SAME: Assertion `Buf[wiID] != 0 && "from assert statement"` failed
// CHECK-NOT:  The test ended.
//
// CHECK-ACC-NOT: {{.*}}assert_in_one_kernel.hpp:10: void kernelFunc(int *, int): {{.*}} [{{[0-3]}},0,0], {{.*}} [0,0,0]
// CHECK-ACC: The test ended.

#include "assert_in_one_kernel.hpp"
