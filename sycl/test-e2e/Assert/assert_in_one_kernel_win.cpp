// https://github.com/intel/llvm/issues/12797
// UNSUPPORTED: windows
// REQUIRES: windows
// RUN: %{build} -DSYCL_FALLBACK_ASSERT=1 -o %t.out
// Shouldn't fail on ACC as fallback assert isn't enqueued there
// RUN: %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt %if acc %{ --check-prefix=CHECK-ACC %}
//
// FIXME Windows version prints '(null)' instead of '<unknown func>' once in a
// while for some insane reason.
// CHECK:      {{.*}}assert_in_one_kernel.hpp:12: {{<unknown func>|(null)}}: {{.*}} [{{[0-3]}},0,0], {{.*}} [0,0,0]
// CHECK-SAME: Assertion `Buf[wiID] != 0 && "from assert statement"` failed.
// CHECK-NOT:  The test ended.
//
// CHECK-ACC-NOT: {{.*}}assert_in_one_kernel.hpp:12: {{<unknown func>|(null)}}: {{.*}} [{{[0-3]}},0,0], {{.*}} [0,0,0]
// CHECK-ACC:  The test ended.

#include "assert_in_one_kernel.hpp"
