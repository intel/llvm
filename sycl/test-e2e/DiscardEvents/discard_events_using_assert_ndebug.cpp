// RUN: %{build} -DNDEBUG -o %t.out
//
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt
//
// The test checks that the last parameter is `nullptr` for
// piEnqueueKernelLaunch.
// {{0|0000000000000000}} is required for various output on Linux and Windows.
//
// CHECK: ---> piEnqueueKernelLaunch(
// CHECK:        pi_event * :
// CHECK-NEXT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// CHECK: The test passed.

#include "discard_events_kernel_using_assert.hpp"
