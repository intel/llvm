// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/raii.h>

struct urEventCreateExpTest : uur::urContextTest {};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEventCreateExpTest);

TEST_P(urEventCreateExpTest, Success) {
  ur_exp_event_desc_t desc = {
      UR_STRUCTURE_TYPE_EXP_EVENT_DESC,
      nullptr,
      static_cast<ur_exp_event_flags_t>(0),
  };

  uur::raii::Event event = nullptr;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventCreateExp(context, device, &desc, event.ptr()));

  if (!event) {
    return;
  }

  ASSERT_NE(*event.ptr(), nullptr);
}

TEST_P(urEventCreateExpTest, SuccessWithProfilingFlag) {
  ur_exp_event_desc_t desc = {
      UR_STRUCTURE_TYPE_EXP_EVENT_DESC,
      nullptr,
      UR_EXP_EVENT_FLAG_ENABLE_PROFILING,
  };

  uur::raii::Event event = nullptr;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventCreateExp(context, device, &desc, event.ptr()));

  if (!event) {
    return;
  }

  ASSERT_NE(*event.ptr(), nullptr);
}

TEST_P(urEventCreateExpTest, InvalidNullHandleContext) {
  ur_exp_event_desc_t desc = {
      UR_STRUCTURE_TYPE_EXP_EVENT_DESC,
      nullptr,
      static_cast<ur_exp_event_flags_t>(0),
  };

  uur::raii::Event event = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urEventCreateExp(nullptr, device, &desc, event.ptr()));
}

TEST_P(urEventCreateExpTest, InvalidNullPointerEventDesc) {
  uur::raii::Event event = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urEventCreateExp(context, device, nullptr, event.ptr()));
}

TEST_P(urEventCreateExpTest, InvalidNullPointerEventHandle) {
  ur_exp_event_desc_t desc = {
      UR_STRUCTURE_TYPE_EXP_EVENT_DESC,
      nullptr,
      static_cast<ur_exp_event_flags_t>(0),
  };

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urEventCreateExp(context, device, &desc, nullptr));
}

TEST_P(urEventCreateExpTest, InvalidNullHandleEventDevice) {
  ur_exp_event_desc_t desc = {
      UR_STRUCTURE_TYPE_EXP_EVENT_DESC,
      nullptr,
      static_cast<ur_exp_event_flags_t>(0),
  };

  uur::raii::Event event = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urEventCreateExp(context, nullptr, &desc, event.ptr()));
}

struct urEnqueueEventsWaitWithBarrierReusableEventTest : uur::urQueueTest {};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(
    urEnqueueEventsWaitWithBarrierReusableEventTest);

TEST_P(urEnqueueEventsWaitWithBarrierReusableEventTest,
       ReusesCallerProvidedEventHandle) {
  ur_exp_event_desc_t desc = {
      UR_STRUCTURE_TYPE_EXP_EVENT_DESC,
      nullptr,
      static_cast<ur_exp_event_flags_t>(0),
  };

  uur::raii::Event signal_event = nullptr;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventCreateExp(context, device, &desc, signal_event.ptr()));
  if (!signal_event) {
    return;
  }

  ur_event_handle_t original = signal_event;
  ur_exp_enqueue_ext_properties_t props = {
      UR_STRUCTURE_TYPE_EXP_ENQUEUE_EXT_PROPERTIES,
      nullptr,
      UR_EXP_ENQUEUE_EXT_FLAG_LOW_POWER_EVENTS_SUPPORT,
  };

  ur_result_t first = urEnqueueEventsWaitWithBarrierExt(
      queue, &props, 0, nullptr, signal_event.ptr());
  if (first == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    return;
  }
  ASSERT_SUCCESS(first);
  ASSERT_EQ(original, signal_event);
  ASSERT_SUCCESS(urEventWait(1, signal_event.ptr()));

  ur_result_t second = urEnqueueEventsWaitWithBarrierExt(
      queue, &props, 0, nullptr, signal_event.ptr());
  ASSERT_SUCCESS(second);
  ASSERT_EQ(original, signal_event);
  ASSERT_SUCCESS(urEventWait(1, signal_event.ptr()));
}
