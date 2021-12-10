// REQUIRES: windows
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -DSYCL_ENABLE_FALLBACK_ASSERT -fsycl -fsycl-targets=%sycl_triple -I %S/Inputs %s %S/Inputs/kernels_in_file2.cpp -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %CPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: %GPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// Shouldn't fail on ACC as fallback assert isn't enqueued there
// RUN: %ACC_RUN_PLACEHOLDER %t.out &> %t.txt
// RUN: %ACC_RUN_PLACEHOLDER FileCheck %s --check-prefix=CHECK-ACC --input-file %t.txt
//
// FIXME Windows version prints '(null)' instead of '<unknown func>' once in a
// while for some insane reason.
// CHECK:      {{.*}}kernels_in_file2.cpp:15: {{<unknown func>|(null)}}: global id: [5,0,0], local id: [1,0,0]
// CHECK-SAME: Assertion `X && "this message from calculus"` failed.
// CHECK-NOT:  this message from file2
// CHECK-NOT:  The test ended.
//
// CHECK-ACC-NOT: {{.*}}kernels_in_file2.cpp:15: {{<unknown func>|(null)}}: global id: [5,0,0], local id: [1,0,0]
// CHECK-ACC: The test ended.

#include "assert_in_multiple_tus.hpp"
