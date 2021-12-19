// FIXME unsupported on level_zero until L0 Plugin support becomes available for
// discard_queue_events
// UNSUPPORTED: level_zero
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -DNDEBUG -o %t.out
//
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out &> %t.txt
// RUN: %CPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out &> %t.txt
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: env SYCL_PI_TRACE=2 %ACC_RUN_PLACEHOLDER %t.out &> %t.txt
// RUN: %ACC_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
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
