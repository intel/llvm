// REQUIRES: windows
// RUN: %{build} -DSYCL_FALLBACK_ASSERT=1 -I %S/Inputs %S/Inputs/kernels_in_file2.cpp -o %t.out
// RUN: %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt
//
// FIXME Windows version prints '(null)' instead of '<unknown func>' once in a
// while for some insane reason.
// CHECK:      {{.*}}kernels_in_file2.cpp:15: {{<unknown func>|\(null\)}}: {{.*}} [5,0,0], {{.*}} [1,0,0]
// CHECK-SAME: Assertion `X && "this message from calculus"` failed.
// CHECK-NOT:  this message from file2
// CHECK-NOT:  The test ended.

#include "assert_in_multiple_tus.hpp"
