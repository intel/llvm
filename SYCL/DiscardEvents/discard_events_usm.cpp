// FIXME unsupported in HIP plugin
// UNSUPPORTED: hip
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
//
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out &> %t.txt
// RUN: %CPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out &> %t.txt
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: env SYCL_PI_TRACE=2 %ACC_RUN_PLACEHOLDER %t.out &> %t.txt
// RUN: %ACC_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
//
// The test checks that the last parameter is `nullptr` for all PI calls that
// should discard events.
// {{0|0000000000000000}} is required for various output on Linux and Windows.
//
// Everything that follows TestQueueOperations()
// CHECK: ---> piextUSMEnqueueMemset(
// CHECK:        pi_event * :
// CHECK-NEXT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// CHECK: ---> piextUSMEnqueueMemcpy(
// CHECK:        pi_event * :
// CHECK-NEXT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// Q.fill don't use piEnqueueMemBufferFill
// CHECK: ---> piEnqueueKernelLaunch(
// CHECK:        pi_event * :
// CHECK-NEXT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// ---> piEnqueueMemBufferCopy(
// CHECK: ---> piextUSMEnqueueMemcpy(
// CHECK:        pi_event * :
// CHECK-NEXT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// CHECK: ---> piextUSMEnqueuePrefetch(
// CHECK:        pi_event * :
// CHECK-NEXT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// CHECK: ---> piextUSMEnqueueMemAdvise(
// CHECK: ) --->  pi_result : PI_SUCCESS
// CHECK-NEXT:         [out]pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// CHECK: ---> piEnqueueEventsWaitWithBarrier(
// CHECK: ) --->  pi_result : PI_SUCCESS
// CHECK-NEXT:         [out]pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// CHECK: ---> piEnqueueKernelLaunch(
// CHECK:        pi_event * :
// CHECK-NEXT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// CHECK: ---> piEnqueueKernelLaunch(
// CHECK:        pi_event * :
// CHECK-NEXT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// RegularQueue
// CHECK: ---> piextUSMEnqueueMemset(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// CHECK: ---> piEnqueueEventsWait(
// CHECK:        pi_event * :
// CHECK-NEXT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// Everything that follows TestQueueOperationsViaSubmit()
// CHECK: ---> piextUSMEnqueueMemset(
// CHECK:        pi_event * :
// CHECK-NEXT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// CHECK: ---> piextUSMEnqueueMemcpy(
// CHECK:        pi_event * :
// CHECK-NEXT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// Q.fill don't use piEnqueueMemBufferFill
// CHECK: ---> piEnqueueKernelLaunch(
// CHECK:        pi_event * :
// CHECK-NEXT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// ---> piEnqueueMemBufferCopy(
// CHECK: ---> piextUSMEnqueueMemcpy(
// CHECK:        pi_event * :
// CHECK-NEXT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// CHECK: ---> piextUSMEnqueuePrefetch(
// CHECK:        pi_event * :
// CHECK-NEXT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// CHECK: ---> piextUSMEnqueueMemAdvise(
// CHECK: ) --->  pi_result : PI_SUCCESS
// CHECK-NEXT:         [out]pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// CHECK: ---> piEnqueueEventsWaitWithBarrier(
// CHECK: ) --->  pi_result : PI_SUCCESS
// CHECK-NEXT:         [out]pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// CHECK: ---> piEnqueueKernelLaunch(
// CHECK:        pi_event * :
// CHECK-NEXT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// CHECK: ---> piEnqueueKernelLaunch(
// CHECK:        pi_event * :
// CHECK-NEXT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// RegularQueue
// CHECK: ---> piextUSMEnqueueMemset(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// CHECK: ---> piEnqueueEventsWait(
// CHECK:        pi_event * :
// CHECK-NEXT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// CHECK: The test passed.

#include "discard_events_test_queue_ops.hpp"
#include <iostream>
int main(int Argc, const char *Argv[]) {

  sycl::property_list Props{
      sycl::property::queue::in_order{},
      sycl::ext::oneapi::property::queue::discard_events{}};
  sycl::queue Q(Props);

  TestQueueOperations(Q);

  TestQueueOperationsViaSubmit(Q);

  std::cout << "The test passed." << std::endl;
  return 0;
}
