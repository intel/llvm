// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include "fixtures.h"
#include <chrono>
#include <thread>
#include <uur/known_failure.h>

struct QueueUSMTestWithParam : uur::IntegrationQueueTestWithParam {
  void SetUp() override {
    UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{},
                         uur::NativeCPU{});

    program_name = "cpy_and_mult_usm";
    UUR_RETURN_ON_FATAL_FAILURE(uur::IntegrationQueueTestWithParam::SetUp());

    ur_device_usm_access_capability_flags_t shared_usm_flags = 0;
    ASSERT_SUCCESS(
        uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_flags));
    if (!(shared_usm_flags & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
      GTEST_SKIP() << "Shared USM is not supported.";
    }

    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                    ArraySize * sizeof(uint32_t), &DeviceMem1));

    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                    ArraySize * sizeof(uint32_t), &DeviceMem2));
  }

  void TearDown() override {
    if (DeviceMem1) {
      ASSERT_SUCCESS(urUSMFree(context, DeviceMem1));
    }
    if (DeviceMem2) {
      ASSERT_SUCCESS(urUSMFree(context, DeviceMem2));
    }
    uur::IntegrationQueueTestWithParam::TearDown();
  }

  void verifyResults(void *DeviceMem, uint32_t ExpectedValue) {
    uint32_t HostMem[ArraySize] = {};
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(Queue, true, HostMem, DeviceMem,
                                      sizeof(uint32_t) * ArraySize, 0, nullptr,
                                      nullptr));

    for (uint32_t i : HostMem) {
      ASSERT_EQ(i, ExpectedValue);
    }
  }

  void *DeviceMem1 = nullptr;
  void *DeviceMem2 = nullptr;
};

UUR_DEVICE_TEST_SUITE_P(
    QueueUSMTestWithParam,
    testing::Values(0, /* In-Order */
                    UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE),
    uur::IntegrationQueueTestWithParam::paramPrinter);

/* Submits multiple kernels that interact with each other by accessing and
 * writing to the same USM memory locations.
 * Checks that when using an IN_ORDER queue, no synchronization is needed
 * between calls to urEnqueueKernelLaunch.
 * Checks that when using an OUT_OF_ORDER queue, synchronizing using only
 * event barriers is enough. */
TEST_P(QueueUSMTestWithParam, QueueUSMTest) {

  std::vector<ur_event_handle_t> EventsFill;
  ur_event_handle_t Event;
  ASSERT_SUCCESS(urEnqueueUSMFill(Queue, DeviceMem1, sizeof(uint32_t),
                                  &InitialValue, ArraySize * sizeof(uint32_t),
                                  0, nullptr, &Event));
  EventsFill.push_back(Event);

  ASSERT_SUCCESS(urEnqueueUSMFill(Queue, DeviceMem2, sizeof(uint32_t),
                                  &InitialValue, ArraySize * sizeof(uint32_t),
                                  0, nullptr, &Event));
  EventsFill.push_back(Event);

  ASSERT_NO_FATAL_FAILURE(submitBarrierIfNeeded(EventsFill));

  constexpr size_t GlobalOffset = 0;
  constexpr size_t NDimensions = 1;
  constexpr uint32_t NumIterations = 5;

  uint32_t CurValueMem1 = InitialValue;
  uint32_t CurValueMem2 = InitialValue;

  std::vector<ur_event_handle_t> EventsKernel;

  for (uint32_t i = 0; i < NumIterations; ++i) {
    /* Copy from DeviceMem2 to DeviceMem1 and multiply by 2 */
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, DeviceMem1));
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 1, nullptr, DeviceMem2));

    ASSERT_SUCCESS(urEnqueueKernelLaunch(Queue, kernel, NDimensions,
                                         &GlobalOffset, &ArraySize, nullptr, 0,
                                         nullptr, &Event));
    ASSERT_NO_FATAL_FAILURE(submitBarrierIfNeeded(Event));

    CurValueMem2 = CurValueMem1 * 2;

    /* Copy from DeviceMem1 to DeviceMem2 and multiply by 2 */
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, DeviceMem2));
    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 1, nullptr, DeviceMem1));

    ASSERT_SUCCESS(urEnqueueKernelLaunch(Queue, kernel, NDimensions,
                                         &GlobalOffset, &ArraySize, nullptr, 0,
                                         nullptr, &Event));
    ASSERT_NO_FATAL_FAILURE(submitBarrierIfNeeded(Event));

    CurValueMem1 = CurValueMem2 * 2;
  }

  ASSERT_SUCCESS(urQueueFinish(Queue));

  ASSERT_NO_FATAL_FAILURE(verifyResults(DeviceMem1, CurValueMem1));
  ASSERT_NO_FATAL_FAILURE(verifyResults(DeviceMem2, CurValueMem2));
}
