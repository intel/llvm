// REQUIRES: windows
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt
//
// FIXME Windows version prints '(null)' instead of '<unknown func>' once in a
// while for some insane reason.
// CHECK:      {{.*}}assert_in_one_kernel.hpp:12: {{<unknown func>|\(null\)}}: {{.*}} [{{[0-3]}},0,0], {{.*}} [0,0,0]
// CHECK-SAME: Assertion `Buf[wiID] != 0 && "from assert statement"` failed.
// CHECK-NOT:  The test ended.

#include "assert_in_one_kernel.hpp"
