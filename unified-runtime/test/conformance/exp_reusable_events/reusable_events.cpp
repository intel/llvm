// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/raii.h>

struct urEventCreateExpTest : uur::urContextTest {};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEventCreateExpTest);

TEST_P(urEventCreateExpTest, Success) {
  ur_exp_event_desc_t desc{
      UR_STRUCTURE_TYPE_EXP_EVENT_DESC,
      nullptr,
      0,
  };

  uur::raii::Event event{};
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventCreateExp(context, device, &desc, event.ptr()));

  if (!event)
    return;

  ASSERT_NE(*event.ptr(), nullptr);
}

TEST_P(urEventCreateExpTest, SuccessWithProfilingFlag) {
  ur_exp_event_desc_t desc = {
      UR_STRUCTURE_TYPE_EXP_EVENT_DESC,
      nullptr,
      UR_EXP_EVENT_FLAG_ENABLE_PROFILING,
  };

  uur::raii::Event event{};
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventCreateExp(context, device, &desc, event.ptr()));

  if (!event)
    return;

  ASSERT_NE(*event.ptr(), nullptr);
}

TEST_P(urEventCreateExpTest, SuccessWithLowPowerSyncMode) {
  ur_exp_event_sync_mode_desc_t sync{
      UR_STRUCTURE_TYPE_EXP_EVENT_SYNC_MODE_DESC,
      nullptr,
      UR_EXP_EVENT_SYNC_MODE_FLAG_LOW_POWER_WAIT,
  };
  ur_exp_event_desc_t desc{
      UR_STRUCTURE_TYPE_EXP_EVENT_DESC,
      &sync,
      0,
  };

  uur::raii::Event event{};
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventCreateExp(context, device, &desc, event.ptr()));

  if (!event)
    return;

  ASSERT_NE(*event.ptr(), nullptr);
}

TEST_P(urEventCreateExpTest, SyncModeFlagsZeroIsNoop) {
  ur_exp_event_sync_mode_desc_t sync{
      UR_STRUCTURE_TYPE_EXP_EVENT_SYNC_MODE_DESC,
      nullptr,
      0,
  };
  ur_exp_event_desc_t desc{
      UR_STRUCTURE_TYPE_EXP_EVENT_DESC,
      &sync,
      0,
  };

  uur::raii::Event event{};
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventCreateExp(context, device, &desc, event.ptr()));
  if (!event)
    return;
  ASSERT_NE(*event.ptr(), nullptr);
}

struct urEnqueueEventsWaitWithBarrierLowPowerEventTest : uur::urQueueTest {};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(
    urEnqueueEventsWaitWithBarrierLowPowerEventTest);

TEST_P(urEnqueueEventsWaitWithBarrierLowPowerEventTest, SignalAndWait) {
  ur_exp_event_sync_mode_desc_t sync{
      UR_STRUCTURE_TYPE_EXP_EVENT_SYNC_MODE_DESC,
      nullptr,
      UR_EXP_EVENT_SYNC_MODE_FLAG_LOW_POWER_WAIT,
  };
  ur_exp_event_desc_t desc{
      UR_STRUCTURE_TYPE_EXP_EVENT_DESC,
      &sync,
      0,
  };

  uur::raii::Event signal_event{};
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventCreateExp(context, device, &desc, signal_event.ptr()));
  if (!signal_event)
    return;

  ur_exp_enqueue_ext_properties_t props{
      UR_STRUCTURE_TYPE_EXP_ENQUEUE_EXT_PROPERTIES,
      nullptr,
      0,
  };

  ur_result_t r = urEnqueueEventsWaitWithBarrierExt(queue, &props, 0, nullptr,
                                                    signal_event.ptr());
  if (r == UR_RESULT_ERROR_UNSUPPORTED_FEATURE)
    return;
  ASSERT_SUCCESS(r);

  ASSERT_SUCCESS(urEventWait(1, signal_event.ptr()));

  ur_event_status_t status = UR_EVENT_STATUS_QUEUED;
  ASSERT_SUCCESS(urEventGetInfo(signal_event,
                                UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                                sizeof(status), &status, nullptr));
  ASSERT_EQ(status, UR_EVENT_STATUS_COMPLETE);
}

TEST_P(urEventCreateExpTest, InvalidNullHandleContext) {
  ur_exp_event_desc_t desc{
      UR_STRUCTURE_TYPE_EXP_EVENT_DESC,
      nullptr,
      0,
  };

  uur::raii::Event event{};
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urEventCreateExp(nullptr, device, &desc, event.ptr()));
}

TEST_P(urEventCreateExpTest, InvalidNullPointerEventDesc) {
  uur::raii::Event event{};
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urEventCreateExp(context, device, nullptr, event.ptr()));
}

TEST_P(urEventCreateExpTest, InvalidNullPointerEventHandle) {
  ur_exp_event_desc_t desc{
      UR_STRUCTURE_TYPE_EXP_EVENT_DESC,
      nullptr,
      0,
  };

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urEventCreateExp(context, device, &desc, nullptr));
}

TEST_P(urEventCreateExpTest, InvalidNullHandleEventDevice) {
  ur_exp_event_desc_t desc{
      UR_STRUCTURE_TYPE_EXP_EVENT_DESC,
      nullptr,
      0,
  };

  uur::raii::Event event{};
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urEventCreateExp(context, nullptr, &desc, event.ptr()));
}

struct urEnqueueEventsWaitWithBarrierReusableEventTest : uur::urQueueTest {};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(
    urEnqueueEventsWaitWithBarrierReusableEventTest);

TEST_P(urEnqueueEventsWaitWithBarrierReusableEventTest,
       ReusesCallerProvidedEventHandle) {
  ur_exp_event_desc_t desc{
      UR_STRUCTURE_TYPE_EXP_EVENT_DESC,
      nullptr,
      0,
  };

  uur::raii::Event signal_event{};
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventCreateExp(context, device, &desc, signal_event.ptr()));

  if (!signal_event)
    return;

  ur_event_handle_t original = signal_event;
  ur_exp_enqueue_ext_properties_t props{
      UR_STRUCTURE_TYPE_EXP_ENQUEUE_EXT_PROPERTIES,
      nullptr,
      0,
  };

  ur_result_t first = urEnqueueEventsWaitWithBarrierExt(
      queue, &props, 0, nullptr, signal_event.ptr());

  if (first == UR_RESULT_ERROR_UNSUPPORTED_FEATURE)
    return;

  ASSERT_SUCCESS(first);
  ASSERT_EQ(original, signal_event);
  ASSERT_SUCCESS(urEventWait(1, signal_event.ptr()));

  ur_result_t second = urEnqueueEventsWaitWithBarrierExt(
      queue, &props, 0, nullptr, signal_event.ptr());

  ASSERT_SUCCESS(second);
  ASSERT_EQ(original, signal_event);
  ASSERT_SUCCESS(urEventWait(1, signal_event.ptr()));
}

TEST_P(urEnqueueEventsWaitWithBarrierReusableEventTest,
       OperationOnDifferentQueueWaitsOnReusableEvent) {
  uur::raii::Queue otherQueue{};
  ASSERT_SUCCESS(urQueueCreate(context, device, nullptr, otherQueue.ptr()));

  ur_exp_event_desc_t desc{UR_STRUCTURE_TYPE_EXP_EVENT_DESC, nullptr, 0};
  uur::raii::Event reusable{};
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventCreateExp(context, device, &desc, reusable.ptr()));

  if (!reusable)
    return;

  ur_exp_enqueue_ext_properties_t props{
      UR_STRUCTURE_TYPE_EXP_ENQUEUE_EXT_PROPERTIES, nullptr, 0};

  ur_result_t result = urEnqueueEventsWaitWithBarrierExt(
      queue, &props, 0, nullptr, reusable.ptr());

  if (result == UR_RESULT_ERROR_UNSUPPORTED_FEATURE)
    return;

  ASSERT_SUCCESS(result);

  ur_event_handle_t deps[] = {reusable};
  uur::raii::Event done{};
  ASSERT_SUCCESS(urEnqueueEventsWaitWithBarrier(
      otherQueue, sizeof(deps) / sizeof(deps[0]), deps, done.ptr()));

  ASSERT_SUCCESS(urQueueFinish(queue));
  ASSERT_SUCCESS(urQueueFinish(otherQueue));
}

struct urEventCreateExpNoPoolingTest : uur::urQueueTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());

    ur_device_usm_access_capability_flags_t deviceUSMSupport = 0;
    ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, deviceUSMSupport));
    if (!deviceUSMSupport) {
      GTEST_SKIP() << "Device USM is not supported";
    }

    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                    allocationSize, &src));
    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                    allocationSize, &dst));
  }

  void TearDown() override {
    if (src) {
      EXPECT_SUCCESS(urUSMFree(context, src));
    }
    if (dst) {
      EXPECT_SUCCESS(urUSMFree(context, dst));
    }
    UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::TearDown());
  }

  // Enqueue enough copies that the device is still busy once this returns,
  // then signal `event` behind them.
  void enqueueLongWorkSignaling(ur_event_handle_t *event) {
    for (size_t i = 0; i < numCopies; ++i) {
      ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, dst, src, allocationSize,
                                        0, nullptr, nullptr));
      ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, src, dst, allocationSize,
                                        0, nullptr, nullptr));
    }
    ur_exp_enqueue_ext_properties_t props{
        UR_STRUCTURE_TYPE_EXP_ENQUEUE_EXT_PROPERTIES, nullptr, 0};
    ASSERT_SUCCESS(
        urEnqueueEventsWaitWithBarrierExt(queue, &props, 0, nullptr, event));
  }

  static constexpr size_t allocationSize = 64 * 1024 * 1024;
  static constexpr size_t numCopies = 64;
  void *src = nullptr;
  void *dst = nullptr;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEventCreateExpNoPoolingTest);

TEST_P(urEventCreateExpNoPoolingTest, Success) {
  ur_exp_event_desc_t desc{
      UR_STRUCTURE_TYPE_EXP_EVENT_DESC,
      nullptr,
      UR_EXP_EVENT_FLAG_NO_POOLING,
  };

  uur::raii::Event event{};
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventCreateExp(context, device, &desc, event.ptr()));
  if (!event)
    return;

  // A never-used event must not report any pending work.
  ur_event_status_t status = UR_EVENT_STATUS_QUEUED;
  ASSERT_SUCCESS(urEventGetInfo(event, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                                sizeof(status), &status, nullptr));
  ASSERT_EQ(status, UR_EVENT_STATUS_COMPLETE);
}

TEST_P(urEventCreateExpNoPoolingTest, SuccessWithProfilingFlag) {
  ur_exp_event_desc_t desc{
      UR_STRUCTURE_TYPE_EXP_EVENT_DESC,
      nullptr,
      UR_EXP_EVENT_FLAG_NO_POOLING | UR_EXP_EVENT_FLAG_ENABLE_PROFILING,
  };

  uur::raii::Event event{};
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventCreateExp(context, device, &desc, event.ptr()));
  if (!event)
    return;

  ur_exp_enqueue_ext_properties_t props{
      UR_STRUCTURE_TYPE_EXP_ENQUEUE_EXT_PROPERTIES, nullptr, 0};
  ur_result_t r =
      urEnqueueEventsWaitWithBarrierExt(queue, &props, 0, nullptr, event.ptr());
  if (r == UR_RESULT_ERROR_UNSUPPORTED_FEATURE)
    return;
  ASSERT_SUCCESS(r);
  ASSERT_SUCCESS(urEventWait(1, event.ptr()));

  uint64_t end = 0;
  ASSERT_SUCCESS(urEventGetProfilingInfo(event, UR_PROFILING_INFO_COMMAND_END,
                                         sizeof(end), &end, nullptr));
  ASSERT_NE(end, 0u);
}

// Releasing a NO_POOLING event while its signal is still pending must not let
// the next NO_POOLING event inherit that pending work: the new event has to be
// backed by a fresh native event, not a recycled one.
TEST_P(urEventCreateExpNoPoolingTest, ReleasedPendingEventIsNotRecycled) {
  ur_exp_event_desc_t desc{
      UR_STRUCTURE_TYPE_EXP_EVENT_DESC,
      nullptr,
      UR_EXP_EVENT_FLAG_NO_POOLING,
  };

  uur::raii::Event first{};
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventCreateExp(context, device, &desc, first.ptr()));
  if (!first)
    return;

  enqueueLongWorkSignaling(first.ptr());
  if (HasFatalFailure())
    return;

  // Drop the only reference while the signal is (most likely) still pending.
  first = nullptr;

  uur::raii::Event second{};
  ASSERT_SUCCESS(urEventCreateExp(context, device, &desc, second.ptr()));

  ur_event_status_t status = UR_EVENT_STATUS_QUEUED;
  ASSERT_SUCCESS(urEventGetInfo(second, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                                sizeof(status), &status, nullptr));
  ASSERT_EQ(status, UR_EVENT_STATUS_COMPLETE);

  // The new event is still fully usable for signaling behind the pending work.
  ur_exp_enqueue_ext_properties_t props{
      UR_STRUCTURE_TYPE_EXP_ENQUEUE_EXT_PROPERTIES, nullptr, 0};
  ASSERT_SUCCESS(urEnqueueEventsWaitWithBarrierExt(queue, &props, 0, nullptr,
                                                   second.ptr()));
  ASSERT_SUCCESS(urEventWait(1, second.ptr()));
  ASSERT_SUCCESS(urQueueFinish(queue));
}

// Without the flag an adapter may hand out a recycled native event; with the
// flag every creation must yield a distinct, never-used event even when the
// previous one is still alive.
TEST_P(urEventCreateExpNoPoolingTest, DistinctWhileAlive) {
  ur_exp_event_desc_t desc{
      UR_STRUCTURE_TYPE_EXP_EVENT_DESC,
      nullptr,
      UR_EXP_EVENT_FLAG_NO_POOLING,
  };

  uur::raii::Event first{};
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventCreateExp(context, device, &desc, first.ptr()));
  if (!first)
    return;

  uur::raii::Event second{};
  ASSERT_SUCCESS(urEventCreateExp(context, device, &desc, second.ptr()));
  ASSERT_NE(first.get(), second.get());

  ur_native_handle_t firstNative = 0, secondNative = 0;
  ASSERT_SUCCESS(urEventGetNativeHandle(first, &firstNative));
  ASSERT_SUCCESS(urEventGetNativeHandle(second, &secondNative));
  ASSERT_NE(firstNative, secondNative);
}
