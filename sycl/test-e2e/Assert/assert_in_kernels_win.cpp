// REQUIRES: windows
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt
//
// CHECK-NOT:  One shouldn't see this message
// FIXME Windows version prints '(null)' instead of '<unknown func>' once in a
// while for some insane reason.
// CHECK:      {{.*}}assert_in_kernels.hpp:27: {{<unknown func>|\(null\)}}: {{.*}} [{{[0,2]}},0,0], {{.*}} [0,0,0]
// CHECK-SAME: Assertion `Buf[wiID] == 0 && "from assert statement"` failed.
// CHECK-NOT:  test aborts earlier, one shouldn't see this message
// CHECK-NOT:  The test ended.

#include "assert_in_kernels.hpp"
