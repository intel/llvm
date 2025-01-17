// Copyright (C) 2022-2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include "device.hpp"
#include "event.hpp"
#include "fixtures.h"
#include "raii.h"

using cudaEventTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(cudaEventTest);

// Testing the urEventGetInfo behaviour for natively constructed (Cuda) events.
// Backend interop APIs can lead to creating event objects that are not fully
// initialized. In the Cuda adapter, an event can have nullptr command queue
// because the interop API does not associate a UR-owned queue with the event.
TEST_P(cudaEventTest, GetQueueFromEventCreatedWithNativeHandle) {
  CUcontext cuda_ctx = device->getNativeContext();
  EXPECT_NE(cuda_ctx, nullptr);
  RAIICUevent cuda_event;
  ASSERT_SUCCESS_CUDA(cuCtxSetCurrent(cuda_ctx));
  ASSERT_SUCCESS_CUDA(cuEventCreate(cuda_event.ptr(), CU_EVENT_DEFAULT));

  auto native_event = reinterpret_cast<ur_native_handle_t>(cuda_event.get());
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
