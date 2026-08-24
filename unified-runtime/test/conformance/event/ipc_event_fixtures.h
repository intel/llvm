// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_CONFORMANCE_EVENT_IPC_EVENT_FIXTURES_H_INCLUDED
#define UR_CONFORMANCE_EVENT_IPC_EVENT_FIXTURES_H_INCLUDED

#include <uur/fixtures.h>

namespace uur {
namespace event {

/// Fixture for the inter-process event sharing experimental APIs
/// (urIPC{Get,Put,Open}EventHandleExp).
///
/// SetUp:
///   - skips on devices that don't advertise
///     UR_DEVICE_INFO_IPC_EVENT_SUPPORT_EXP,
///   - creates an IPC-shareable event via urEventCreateExp with
///     UR_EXP_EVENT_FLAG_IPC_EXP set, and signals it via the reusable-events
///     API (urEnqueueEventsWaitWithBarrierExt + urQueueFinish).
/// Derives from urQueueTest (no profiling) since IPC and per-event profiling
/// are mutually exclusive.
struct urIPCEventTest : uur::urQueueTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::SetUp());

    ur_bool_t ipcEventSupport = false;
    ur_result_t queryResult =
        urDeviceGetInfo(device, UR_DEVICE_INFO_IPC_EVENT_SUPPORT_EXP,
                        sizeof(ipcEventSupport), &ipcEventSupport, nullptr);
    // Adapters that don't implement the query report it as an unsupported
    // enumeration or feature; either way the feature is unavailable and the
    // test is skipped rather than failed.
    if (queryResult == UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION ||
        queryResult == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      GTEST_SKIP() << "IPC event feature is not supported on this device.";
    }
    ASSERT_SUCCESS(queryResult);
    if (!ipcEventSupport) {
      GTEST_SKIP() << "IPC event feature is not supported on this device.";
    }

    ur_exp_event_desc_t desc{UR_STRUCTURE_TYPE_EXP_EVENT_DESC, nullptr,
                             UR_EXP_EVENT_FLAG_IPC_EXP};
    ASSERT_SUCCESS(urEventCreateExp(context, device, &desc, &event));
    ASSERT_NE(event, nullptr);

    // Signal the producer event through the reusable-events API:
    // urEnqueueEventsWaitWithBarrierExt reuses it as the signal event and
    // urQueueFinish drains the queue so the event is already signaled when the
    // consumer side waits on the imported handle. This keeps the test
    // backend-agnostic (no direct Level Zero calls).
    ur_exp_enqueue_ext_properties_t props{
        UR_STRUCTURE_TYPE_EXP_ENQUEUE_EXT_PROPERTIES, nullptr, 0};
    ASSERT_SUCCESS(
        urEnqueueEventsWaitWithBarrierExt(queue, &props, 0, nullptr, &event));
    ASSERT_SUCCESS(urQueueFinish(queue));
  }

  void TearDown() override {
    if (event) {
      EXPECT_SUCCESS(urEventRelease(event));
      event = nullptr;
    }
    uur::urQueueTest::TearDown();
  }

  ur_event_handle_t event = nullptr;
};

} // namespace event
} // namespace uur

#endif // UR_CONFORMANCE_EVENT_IPC_EVENT_FIXTURES_H_INCLUDED
