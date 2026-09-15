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
