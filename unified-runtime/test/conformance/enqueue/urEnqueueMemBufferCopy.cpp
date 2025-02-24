// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "helpers.h"
#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urEnqueueMemBufferCopyTestWithParam : uur::urQueueTestWithParam<size_t> {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urQueueTestWithParam::SetUp());
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_WRITE_ONLY, size,
                                     nullptr, &src_buffer));
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_ONLY, size,
                                     nullptr, &dst_buffer));
    input.assign(count, 42);
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, src_buffer, true, 0, size,
                                           input.data(), 0, nullptr, nullptr));
  }

  void TearDown() override {
    if (src_buffer) {
      EXPECT_SUCCESS(urMemRelease(src_buffer));
    }
    if (src_buffer) {
      EXPECT_SUCCESS(urMemRelease(dst_buffer));
    }
    urQueueTestWithParam::TearDown();
  }

  const size_t count = std::get<1>(this->GetParam());
  const size_t size = sizeof(uint32_t) * count;
  ur_mem_handle_t src_buffer = nullptr;
  ur_mem_handle_t dst_buffer = nullptr;
  std::vector<uint32_t> input;
};

static std::vector<size_t> test_parameters{1024, 2500, 4096, 6000};

UUR_DEVICE_TEST_SUITE_WITH_PARAM(urEnqueueMemBufferCopyTestWithParam,
                                 ::testing::ValuesIn(test_parameters),
                                 uur::deviceTestWithParamPrinter<size_t>);

TEST_P(urEnqueueMemBufferCopyTestWithParam, Success) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  ASSERT_SUCCESS(urEnqueueMemBufferCopy(queue, src_buffer, dst_buffer, 0, 0,
                                        size, 0, nullptr, nullptr));
  std::vector<uint32_t> output(count, 1);
  ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, dst_buffer, true, 0, size,
                                        output.data(), 0, nullptr, nullptr));
  ASSERT_EQ(input, output);
}

TEST_P(urEnqueueMemBufferCopyTestWithParam, InvalidNullHandleQueue) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urEnqueueMemBufferCopy(nullptr, src_buffer, dst_buffer, 0, 0,
                                          size, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferCopyTestWithParam, InvalidNullHandleBufferSrc) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urEnqueueMemBufferCopy(queue, nullptr, dst_buffer, 0, 0,
                                          size, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferCopyTestWithParam, InvalidNullHandleBufferDst) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urEnqueueMemBufferCopy(queue, src_buffer, nullptr, 0, 0,
                                          size, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferCopyTestWithParam, InvalidNullPtrEventWaitList) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  ASSERT_EQ_RESULT(urEnqueueMemBufferCopy(queue, src_buffer, dst_buffer, 0, 0,
                                          size, 1, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

  ur_event_handle_t validEvent;
  ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

  ASSERT_EQ_RESULT(urEnqueueMemBufferCopy(queue, src_buffer, dst_buffer, 0, 0,
                                          size, 0, &validEvent, nullptr),
                   UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

  ur_event_handle_t inv_evt = nullptr;
  ASSERT_EQ_RESULT(urEnqueueMemBufferCopy(queue, src_buffer, dst_buffer, 0, 0,
                                          size, 1, &inv_evt, nullptr),
                   UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

  ASSERT_SUCCESS(urEventRelease(validEvent));
}

TEST_P(urEnqueueMemBufferCopyTestWithParam, InvalidSize) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urEnqueueMemBufferCopy(queue, src_buffer, dst_buffer, 1, 0,
                                          size, 0, nullptr, nullptr));
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urEnqueueMemBufferCopy(queue, src_buffer, dst_buffer, 0, 1,
                                          size, 0, nullptr, nullptr));
}

using urEnqueueMemBufferCopyMultiDeviceTest =
    uur::urMultiDeviceMemBufferQueueTest;
UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urEnqueueMemBufferCopyMultiDeviceTest);

TEST_P(urEnqueueMemBufferCopyMultiDeviceTest, CopyReadDifferentQueues) {
  // First queue does a fill.
  const uint32_t input = 42;
  ASSERT_SUCCESS(urEnqueueMemBufferFill(
      queues[0], buffer, &input, sizeof(input), 0, size, 0, nullptr, nullptr));

  // Then a copy.
  ur_mem_handle_t dst_buffer = nullptr;
  ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_ONLY, size,
                                   nullptr, &dst_buffer));
  EXPECT_SUCCESS(urEnqueueMemBufferCopy(queues[0], buffer, dst_buffer, 0, 0,
                                        size, 0, nullptr, nullptr));

  // Wait for the queue to finish executing.
  EXPECT_SUCCESS(urEnqueueEventsWait(queues[0], 0, nullptr, nullptr));

  // Then the remaining queues do blocking reads from the buffer. Since the
  // queues target different devices this checks that any devices memory has
  // been synchronized.
  for (unsigned i = 1; i < queues.size(); ++i) {
    const auto queue = queues[i];
    std::vector<uint32_t> output(count, 0);
    EXPECT_SUCCESS(urEnqueueMemBufferRead(queue, dst_buffer, true, 0, size,
                                          output.data(), 0, nullptr, nullptr));
    for (unsigned j = 0; j < count; ++j) {
      EXPECT_EQ(input, output[j])
          << "Result on queue " << i << " did not match at index " << j << "!";
    }
  }

  EXPECT_SUCCESS(urMemRelease(dst_buffer));
}
