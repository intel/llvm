// REQUIRES: windows
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -DSYCL_ENABLE_FALLBACK_ASSERT -fsycl -fsycl-targets=%sycl_triple %s -o %t.out %threads_lib
// RUN: %CPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %CPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
//
// Since this is a multi-threaded application enable memory tracking and
// deferred release feature in the Level Zero plugin to avoid releasing memory
// too early. This is necessary because currently SYCL RT sets indirect access
// flag for all kernels and the Level Zero runtime doesn't support deferred
// release yet.
// RUN: env SYCL_PI_LEVEL_ZERO_TRACK_INDIRECT_ACCESS_MEMORY=1 %GPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// Shouldn't fail on ACC as fallback assert isn't enqueued there
// RUN: %ACC_RUN_PLACEHOLDER %t.out &> %t.txt
// RUN: %ACC_RUN_PLACEHOLDER FileCheck %s --check-prefix=CHECK-ACC --input-file %t.txt
//
// FIXME Windows version prints '(null)' instead of '<unknown func>' once in a
// while for some insane reason.
// CHECK:      {{.*}}assert_in_simultaneous_kernels.hpp:12: {{<unknown func>|(null)}}: global id: [9,7,0], local id: [0,0,0]
// CHECK-SAME: Assertion `false && "from assert statement"` failed.
// CHECK-NOT:  The test ended.
//
// CHECK-ACC-NOT: {{.*}}assert_in_simultaneous_kernels.hpp:12: {{<unknown func>|(null)}}: global id: [9,7,0], local id: [0,0,0]
// CHECK-ACC:  The test ended.

#include "assert_in_simultaneous_kernels.hpp"
