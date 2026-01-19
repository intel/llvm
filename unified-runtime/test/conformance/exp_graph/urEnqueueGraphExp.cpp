// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include "uur/raii.h"

#include <vector>

struct urEnqueueGraphExpTest : uur::urGraphExecutableExpTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urGraphExecutableExpTest::SetUp());

    // TODO: Re-enable urEnqueueGraph tests on L0V2.
    // See: https://github.com/intel/llvm/issues/20884.
    UUR_KNOWN_FAILURE_ON(uur::LevelZeroV2{});
  }
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEnqueueGraphExpTest);

TEST_P(urEnqueueGraphExpTest, Success) {
  ASSERT_NO_FATAL_FAILURE(verifyData(false));
  ASSERT_SUCCESS(urEnqueueGraphExp(queue, exGraph, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  ASSERT_NO_FATAL_FAILURE(verifyData(true));
}

TEST_P(urEnqueueGraphExpTest, SuccessWithEvent) {
  uur::raii::Event graphEvent = nullptr;
  ASSERT_SUCCESS(
      urEnqueueGraphExp(queue, exGraph, 0, nullptr, graphEvent.ptr()));
  ASSERT_SUCCESS(urQueueFlush(queue));
  ASSERT_SUCCESS(urEventWait(1, graphEvent.ptr()));
}

TEST_P(urEnqueueGraphExpTest, InvalidNullHandleQueue) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urEnqueueGraphExp(nullptr, exGraph, 0, nullptr, nullptr));
}

TEST_P(urEnqueueGraphExpTest, InvalidNullHandleExGraph) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urEnqueueGraphExp(queue, nullptr, 0, nullptr, nullptr));
}

TEST_P(urEnqueueGraphExpTest, InvalidEventWaitListArray) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                   urEnqueueGraphExp(queue, exGraph, 1, nullptr, nullptr));
}

TEST_P(urEnqueueGraphExpTest, InvalidEventWaitListSize) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                   urEnqueueGraphExp(queue, exGraph, 0,
                                     (ur_event_handle_t *)0xC0FFEE, nullptr));
}

TEST_P(urEnqueueGraphExpTest, SuccessMultipleExecutions) {
  const size_t numExecutions = 5;

  for (size_t i = 0; i < numExecutions; ++i) {
    ASSERT_NO_FATAL_FAILURE(verifyData(false));

    ASSERT_SUCCESS(urEnqueueGraphExp(queue, exGraph, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    ASSERT_NO_FATAL_FAILURE(verifyData(true));
    ASSERT_NO_FATAL_FAILURE(resetData());
  }
}

TEST_P(urEnqueueGraphExpTest, SuccessEventDependant) {
  uur::raii::Event fillEvent1 = nullptr;
  uur::raii::Event fillEvent2 = nullptr;
  uur::raii::Event graphEvent = nullptr;

  std::vector<uint8_t> pattern2 = std::vector<uint8_t>(patternSize);
  uur::generateMemFillPattern(pattern2);
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, deviceMem, patternSize,
                                  pattern2.data(), allocationSize / 2, 0,
                                  nullptr, fillEvent1.ptr()));
  ASSERT_SUCCESS(urEnqueueUSMFill(
      queue, static_cast<uint8_t *>(deviceMem) + allocationSize / 2,
      patternSize, pattern2.data(), allocationSize / 2, 0, nullptr,
      fillEvent2.ptr()));

  ur_event_handle_t waitEvents[] = {fillEvent1.get(), fillEvent2.get()};
  ASSERT_SUCCESS(
      urEnqueueGraphExp(queue, exGraph, 2, waitEvents, graphEvent.ptr()));

  ASSERT_SUCCESS(urEventWait(1, graphEvent.ptr()));

  ASSERT_NO_FATAL_FAILURE(verifyData(true));
}

TEST_P(urEnqueueGraphExpTest, SuccessEventOrdering) {
  uur::raii::Event clearEvent = nullptr;
  uur::raii::Event graphEvent = nullptr;
  uur::raii::Event verifyEvent = nullptr;

  const uint8_t zero = 0;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, deviceMem, sizeof(zero), &zero,
                                  allocationSize, 0, nullptr,
                                  clearEvent.ptr()));

  ASSERT_SUCCESS(
      urEnqueueGraphExp(queue, exGraph, 1, clearEvent.ptr(), graphEvent.ptr()));

  ASSERT_SUCCESS(urEnqueueUSMFill(queue, deviceMem, sizeof(zero), &zero,
                                  allocationSize, 1, graphEvent.ptr(),
                                  verifyEvent.ptr()));

  ASSERT_SUCCESS(urEventWait(1, verifyEvent.ptr()));

  ASSERT_NO_FATAL_FAILURE(verifyData(false));
}
