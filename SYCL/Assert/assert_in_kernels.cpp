// REQUIRES: linux
// RUN: %clangxx -DSYCL_FALLBACK_ASSERT=1 -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %CPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: %GPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// Shouldn't fail on ACC as fallback assert isn't enqueued there
// RUN: %ACC_RUN_PLACEHOLDER %t.out &> %t.txt
// RUN: %ACC_RUN_PLACEHOLDER FileCheck %s --check-prefix=CHECK-ACC --input-file %t.txt
//
// CHECK-NOT:  One shouldn't see this message
// CHECK:      {{.*}}assert_in_kernels.hpp:25: void kernelFunc2(int *, int): {{.*}} [{{[0,2]}},0,0], {{.*}} [0,0,0]
// CHECK-SAME: Assertion `Buf[wiID] == 0 && "from assert statement"` failed.
// CHECK-NOT:  test aborts earlier, one shouldn't see this message
// CHECK-NOT:  The test ended.
//
// CHECK-ACC-NOT: {{.*}}assert_in_kernels.hpp:25: void kernelFunc2(int *, int): {{.*}} [{{[0,2]}},0,0], {{.*}} [0,0,0]
// CHECK-ACC: The test ended.

#include "assert_in_kernels.hpp"
