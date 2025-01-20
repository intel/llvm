// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "helpers.h"
#include <uur/fixtures.h>
#include <uur/known_failure.h>

using urEnqueueMemBufferWriteTestWithParam =
    uur::urMemBufferQueueTestWithParam<uur::mem_buffer_test_parameters_t>;

UUR_DEVICE_TEST_SUITE_P(
    urEnqueueMemBufferWriteTestWithParam,
    ::testing::ValuesIn(uur::mem_buffer_test_parameters),
    uur::printMemBufferTestString<urEnqueueMemBufferWriteTestWithParam>);

TEST_P(urEnqueueMemBufferWriteTestWithParam, Success) {
  std::vector<uint32_t> input(count, 42);
  ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                         input.data(), 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferWriteTestWithParam, SuccessWriteRead) {
  std::vector<uint32_t> input(count, 42);
  ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                         input.data(), 0, nullptr, nullptr));
  std::vector<uint32_t> output(count, 0);
  ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, 0, size,
                                        output.data(), 0, nullptr, nullptr));
  for (size_t index = 0; index < count; index++) {
    ASSERT_EQ(input[index], output[index]);
  }
}

TEST_P(urEnqueueMemBufferWriteTestWithParam, InvalidNullHandleQueue) {
  std::vector<uint32_t> input(count, 42);
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urEnqueueMemBufferWrite(nullptr, buffer, true, 0, size,
                                           input.data(), 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferWriteTestWithParam, InvalidNullHandleBuffer) {
  std::vector<uint32_t> input(count, 42);
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urEnqueueMemBufferWrite(queue, nullptr, true, 0, size,
                                           input.data(), 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferWriteTestWithParam, InvalidNullPointerSrc) {
  std::vector<uint32_t> input(count, 42);
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                           nullptr, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemBufferWriteTestWithParam, InvalidNullPtrEventWaitList) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  std::vector<uint32_t> input(count, 42);
  ASSERT_EQ_RESULT(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                           input.data(), 1, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

  ur_event_handle_t validEvent;
  ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

  ASSERT_EQ_RESULT(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                           input.data(), 0, &validEvent,
                                           nullptr),
                   UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

  ur_event_handle_t inv_evt = nullptr;
  ASSERT_EQ_RESULT(urEnqueueMemBufferWrite(queue, buffer, true, 0, size,
                                           input.data(), 1, &inv_evt, nullptr),
                   UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

  ASSERT_SUCCESS(urEventRelease(validEvent));
}

TEST_P(urEnqueueMemBufferWriteTestWithParam, InvalidSize) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  std::vector<uint32_t> output(count, 42);
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urEnqueueMemBufferWrite(queue, buffer, true, 1, size,
                                           output.data(), 0, nullptr, nullptr));
}
