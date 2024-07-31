// RUN: %{build} -o %t.out
//
// On level_zero Q.fill uses urEnqueueKernelLaunch and not urEnqueueUSMFill
// due to https://github.com/intel/llvm/issues/13787
//
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt --check-prefixes=CHECK%if level_zero %{,CHECK-L0%} %else %{,CHECK-OTHER%}
//
// REQUIRES: aspect-usm_shared_allocations
// The test checks that the last parameter is `nullptr` for all UR calls that
// should discard events.
// {{0|0000000000000000}} is required for various output on Linux and Windows.
// NOTE: urEnqueueUSMPrefetch and urEnqueueUSMAdvise in the CUDA and
//       HIP backends may return a warning result on Windows with error-code
//       66 (UR_RESULT_ERROR_ADAPTER_SPECIFIC) if USM managed memory is not
//       supported or if unsupported advice flags are used for the latter API.
//       Since it is a warning it is safe to ignore for this test.
//
// Everything that follows TestQueueOperations()
// CHECK: ---> urEnqueueUSMFill(
// CHECK-SAME: .phEvent = nullptr
//
// CHECK: ---> urEnqueueUSMMemcpy(
// CHECK-SAME: .phEvent = nullptr
//
// Level-zero backend doesn't use urEnqueueUSMFill
// CHECK-L0: ---> urEnqueueKernelLaunch({{.*}} .phEvent = nullptr
// CHECK-OTHER: ---> urEnqueueUSMFill({{.*}} .phEvent = nullptr
//
// ---> urEnqueueUSMMemcpy(
// CHECK: ---> urEnqueueUSMMemcpy(
// CHECK-SAME: .phEvent = nullptr
//
// CHECK: ---> urEnqueueUSMPrefetch(
// CHECK-SAME: .phEvent = nullptr
//
// CHECK: ---> urEnqueueUSMAdvise(
// CHECK-SAME: .phEvent = nullptr
// CHECK-SAME: ) -> {{UR_RESULT_SUCCESS|UR_RESULT_ERROR_ADAPTER_SPECIFIC}}
//
// CHECK: ---> urEnqueueKernelLaunch(
// CHECK-SAME: .phEvent = nullptr
//
// CHECK: ---> urEnqueueKernelLaunch(
// CHECK-SAME: .phEvent = nullptr
//
// RegularQueue
// CHECK-NOT: ---> urEnqueueUSMFill({{.*}} .phEvent = nullptr
// CHECK: ---> urEnqueueUSMFill(
// CHECK-SAME: -> UR_RESULT_SUCCESS
//
// CHECK: ---> urEnqueueEventsWait(
// CHECK-SAME: .phEvent = nullptr
//
// Everything that follows TestQueueOperationsViaSubmit()
// CHECK: ---> urEnqueueUSMFill(
// CHECK-SAME: .phEvent = nullptr
//
// CHECK: ---> urEnqueueUSMMemcpy(
// CHECK-SAME: .phEvent = nullptr
//
// Level-zero backend doesn't use urEnqueueUSMFill
// CHECK-L0: ---> urEnqueueKernelLaunch({{.*}} .phEvent = nullptr
// CHECK-OTHER: ---> urEnqueueUSMFill({{.*}} .phEvent = nullptr
//
// ---> urEnqueueUSMMemcpy(
// CHECK: ---> urEnqueueUSMMemcpy(
// CHECK-SAME: .phEvent = nullptr
//
// CHECK: ---> urEnqueueUSMPrefetch(
// CHECK-SAME: .phEvent = nullptr
//
// CHECK: ---> urEnqueueUSMAdvise(
// CHECK-SAME: .phEvent = nullptr
// CHECK-SAME: ) -> {{UR_RESULT_SUCCESS|UR_RESULT_ERROR_ADAPTER_SPECIFIC}}
//
// CHECK: ---> urEnqueueKernelLaunch(
// CHECK-SAME: .phEvent = nullptr
//
// CHECK: ---> urEnqueueKernelLaunch(
// CHECK-SAME: .phEvent = nullptr
//
// RegularQueue
// CHECK-NOT: ---> urEnqueueUSMFill({{.*}} .phEvent = nullptr
// CHECK: ---> urEnqueueUSMFill(
// CHECK-SAME: -> UR_RESULT_SUCCESS
//
// CHECK: ---> urEnqueueEventsWait(
// CHECK-SAME: .phEvent = nullptr
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
