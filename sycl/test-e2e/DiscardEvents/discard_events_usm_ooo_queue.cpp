// REQUIRES: usm_shared_allocations
// RUN: %{build} -o %t.out

// RUN: env SYCL_PI_TRACE=2 %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt

// The test checks that the last parameter is not `nullptr` for all PI calls
// that should discard events.
// {{0|0000000000000000}} is required for various output on Linux and Windows.
// NOTE: piextUSMEnqueuePrefetch and piextUSMEnqueueMemAdvise in the CUDA and
//       HIP backends may return a warning result on Windows with error-code
//       -996 (PI_ERROR_PLUGIN_SPECIFIC_ERROR) if USM managed memory is not
//       supported or if unsupported advice flags are used for the latter API.
//       Since it is a warning it is safe to ignore for this test.
//
// Everything that follows TestQueueOperations()
// CHECK: ---> piextUSMEnqueueMemset(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// CHECK: ---> piextUSMEnqueueMemcpy(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// Q.fill don't use piEnqueueMemBufferFill
// CHECK: ---> piEnqueueKernelLaunch(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// ---> piEnqueueMemBufferCopy(
// CHECK: ---> piextUSMEnqueueMemcpy(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// CHECK: ---> piextUSMEnqueuePrefetch(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : {{PI_SUCCESS|-996}}
//
// CHECK: ---> piextUSMEnqueueMemAdvise(
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : {{PI_SUCCESS|-996}}
//
// CHECK: ---> piEnqueueEventsWaitWithBarrier(
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// CHECK: ---> piEnqueueKernelLaunch(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// CHECK: ---> piEnqueueKernelLaunch(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// RegularQueue
// CHECK: ---> piextUSMEnqueueMemset(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// CHECK: ---> piEnqueueEventsWait(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// Everything that follows TestQueueOperationsViaSubmit()
// CHECK: ---> piextUSMEnqueueMemset(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// CHECK: ---> piextUSMEnqueueMemcpy(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// Q.fill don't use piEnqueueMemBufferFill
// CHECK: ---> piEnqueueKernelLaunch(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// ---> piEnqueueMemBufferCopy(
// CHECK: ---> piextUSMEnqueueMemcpy(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// CHECK: ---> piextUSMEnqueuePrefetch(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : {{PI_SUCCESS|-996}}
//
// CHECK: ---> piextUSMEnqueueMemAdvise(
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : {{PI_SUCCESS|-996}}
//
// CHECK: ---> piEnqueueEventsWaitWithBarrier(
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// CHECK: ---> piEnqueueKernelLaunch(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// CHECK: ---> piEnqueueKernelLaunch(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// RegularQueue
// CHECK: ---> piextUSMEnqueueMemset(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// CHECK: ---> piEnqueueEventsWait(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// CHECK: The test passed.

#include "discard_events_test_queue_ops.hpp"
#include <iostream>

int main(int Argc, const char *Argv[]) {

  sycl::property_list Props{
      sycl::ext::oneapi::property::queue::discard_events{}};
  sycl::queue OOO_Q(Props);

  TestQueueOperations(OOO_Q);

  TestQueueOperationsViaSubmit(OOO_Q);

  std::cout << "The test passed." << std::endl;
  return 0;
}
