// FIXME unsupported on CUDA and HIP until fallback libdevice becomes available
// UNSUPPORTED: cuda || hip
//
// UNSUPPORTED: ze_debug-1,ze_debug4
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
//
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out &> %t.txt
// RUN: %CPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out &> %t.txt
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: env SYCL_PI_TRACE=2 %ACC_RUN_PLACEHOLDER %t.out &> %t.txt
// RUN: %ACC_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
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
