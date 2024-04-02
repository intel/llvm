// FIXME unsupported on CUDA and HIP until fallback libdevice becomes available
// UNSUPPORTED: cuda || hip
//
// UNSUPPORTED: ze_debug
// RUN: %{build} -o %t.out
//
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt
//
// The test checks that the last parameter is not `nullptr` for
// piEnqueueKernelLaunch.
// {{0|0000000000000000}} is required for various output on Linux and Windows.
//
// CHECK: ---> piEnqueueKernelLaunch(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// CHECK: The test passed.

#include "discard_events_kernel_using_assert.hpp"
