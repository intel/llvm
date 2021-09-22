// REQUIRES: windows
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out %threads_lib
// RUN: %CPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %CPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: %GPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: %ACC_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %ACC_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
//
// FIXME Windows versionprints '(null)' instead of '<unknown func>' once in a
// while for some insane reason.
// CHECK:      {{.*}}assert_in_simultaneous_kernels.hpp:12: {{<unknown func>|(null)}}: global id: [9,7,0], local id: [0,0,0]
// CHECK-SAME: Assertion `false && "from assert statement"` failed.
// CHECK-NOT:  The test ended.

#include "assert_in_simultaneous_kernels.hpp"
