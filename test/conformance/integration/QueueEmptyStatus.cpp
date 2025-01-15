// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include <chrono>
#include <thread>
#include <uur/known_failure.h>

struct QueueEmptyStatusTestWithParam : uur::IntegrationQueueTestWithParam {

  void SetUp() override {
    UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{},
                         uur::NativeCPU{});

    program_name = "multiply";
    UUR_RETURN_ON_FATAL_FAILURE(uur::IntegrationQueueTestWithParam::SetUp());

    ur_device_usm_access_capability_flags_t shared_usm_flags = 0;
    ASSERT_SUCCESS(
        uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_flags));
    if (!(shared_usm_flags & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
      GTEST_SKIP() << "Shared USM is not supported.";
    }

    // QUEUE_INFO_EMPTY isn't supported by all adapters
    ur_bool_t empty_check = false;
    auto result = urQueueGetInfo(queue, UR_QUEUE_INFO_EMPTY,
                                 sizeof(empty_check), &empty_check, nullptr);
    if (result == UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION) {
      GTEST_SKIP() << "QUEUE_INFO_EMPTY is not supported.";
    }
    ASSERT_SUCCESS(result);

    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                    ArraySize * sizeof(uint32_t), &SharedMem));
  }

  void TearDown() override {
    if (SharedMem) {
      ASSERT_SUCCESS(urUSMFree(context, SharedMem));
    }
    uur::IntegrationQueueTestWithParam::TearDown();
  }

  void submitWorkToQueue() {
    ur_event_handle_t Event;
    ASSERT_SUCCESS(urEnqueueUSMFill(Queue, SharedMem, sizeof(uint32_t),
                                    &InitialValue, ArraySize * sizeof(uint32_t),
                                    0, nullptr, &Event));
    ASSERT_NO_FATAL_FAILURE(submitBarrierIfNeeded(Event));

    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, SharedMem));

    constexpr size_t global_offset = 0;
    constexpr size_t n_dimensions = 1;
    constexpr uint32_t num_iterations = 5;
    for (uint32_t i = 0; i < num_iterations; ++i) {
      ASSERT_SUCCESS(urEnqueueKernelLaunch(Queue, kernel, n_dimensions,
                                           &global_offset, &ArraySize, nullptr,
                                           0, nullptr, &Event));
      ASSERT_NO_FATAL_FAILURE(submitBarrierIfNeeded(Event));
    }

    ASSERT_SUCCESS(urQueueFlush(Queue));
  }

  void waitUntilQueueEmpty() const {

    using namespace std::chrono_literals;

    constexpr auto step = 500ms;
    constexpr auto maxWait = 5000ms;

    /* Wait a bit until work finishes running. We don't synchronize with
     * urQueueFinish() because we want to check if the status is set without
     * calling it explicitly. */
    for (auto currentWait = 0ms; currentWait < maxWait; currentWait += step) {
      std::this_thread::sleep_for(step);

      ur_bool_t is_queue_empty;
      ASSERT_SUCCESS(urQueueGetInfo(Queue, UR_QUEUE_INFO_EMPTY,
                                    sizeof(ur_bool_t), &is_queue_empty,
                                    nullptr));
      if (is_queue_empty) {
        return;
      }
    }

    /* If we are here, the test failed. Let's call queue finish to avoid
     * issues when freeing memory */
    ASSERT_SUCCESS(urQueueFinish(Queue));
    GTEST_FAIL();
  }

  void *SharedMem = nullptr;
};

UUR_DEVICE_TEST_SUITE_P(
    QueueEmptyStatusTestWithParam,
    testing::Values(0, /* In-Order */
                    UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE),
    uur::IntegrationQueueTestWithParam::paramPrinter);

/* Submits kernels that have a dependency on each other and checks that the
 * queue submits all the work in the correct order to the device.
 * Explicit synchronization (except for barriers) is avoided in these tests to
 * check that the properties of In-Order and OutOfOrder queues are working as
 * expected */
TEST_P(QueueEmptyStatusTestWithParam, QueueEmptyStatusTest) {
  ASSERT_NO_FATAL_FAILURE(submitWorkToQueue());
  ASSERT_NO_FATAL_FAILURE(waitUntilQueueEmpty());

  constexpr size_t expected_value = 3200;
  for (uint32_t i = 0; i < ArraySize; ++i) {
    ASSERT_EQ(reinterpret_cast<uint32_t *>(SharedMem)[i], expected_value);
  }
}
