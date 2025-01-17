// Copyright (C) 2022-2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include "event.hpp"
#include "fixtures.h"
#include "uur/raii.h"

#include <hip/hip_runtime.h>
#include <tuple>

struct RAIIHipEvent {
  hipEvent_t handle = nullptr;

  ~RAIIHipEvent() {
    if (handle) {
      std::ignore = hipEventDestroy(handle);
    }
  }

  hipEvent_t *ptr() { return &handle; }
  hipEvent_t get() { return handle; }
};

using urHipEventTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urHipEventTest);

// Testing the urEventGetInfo behaviour for natively constructed (HIP) events.
// Backend interop APIs can lead to creating event objects that are not fully
// initialized. In the Cuda adapter, an event can have nullptr command queue
// because the interop API does not associate a UR-owned queue with the event.
TEST_P(urHipEventTest, GetQueueFromEventCreatedWithNativeHandle) {
  RAIIHipEvent hip_event;
  ASSERT_SUCCESS_HIP(hipEventCreateWithFlags(hip_event.ptr(), hipEventDefault));

  auto native_event = reinterpret_cast<ur_native_handle_t>(hip_event.get());
  uur::raii::Event event{nullptr};
  ASSERT_SUCCESS(urEventCreateWithNativeHandle(native_event, context, nullptr,
                                               event.ptr()));
  EXPECT_NE(event, nullptr);

  size_t ret_size{};
  ur_queue_handle_t q{};
  ASSERT_EQ_RESULT(urEventGetInfo(event, UR_EVENT_INFO_COMMAND_QUEUE,
                                  sizeof(ur_queue_handle_t), &q, &ret_size),
                   UR_RESULT_ERROR_ADAPTER_SPECIFIC);
}
