// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urEventWaitTest : uur::urDeviceTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urDeviceTest::SetUp());

    for (size_t i = 0; i < maxNumContexts; ++i) {
      ur_context_handle_t context = nullptr;
      ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, &context));
      ASSERT_NE(context, nullptr);
      contexts.push_back(context);

      ur_queue_handle_t queue = nullptr;
      ASSERT_SUCCESS(urQueueCreate(context, device, 0, &queue));
      ASSERT_NE(queue, nullptr);
      queues.push_back(queue);

      src_buffer.emplace_back();
      dst_buffer.emplace_back();

      ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, size,
                                       nullptr, &src_buffer[i]));
      ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, size,
                                       nullptr, &dst_buffer[i]));
      input.emplace_back();
      input[i].assign(count, uint32_t(99 + i));
      ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, src_buffer[i], true, 0,
                                             size, input[i].data(), 0, nullptr,
                                             nullptr));
    }
  }

  void TearDown() override {
    for (size_t i = 0; i < src_buffer.size(); ++i) {
      EXPECT_SUCCESS(urMemRelease(src_buffer[i]));
      EXPECT_SUCCESS(urMemRelease(dst_buffer[i]));
    }
    for (size_t i = 0; i < queues.size(); ++i) {
      EXPECT_SUCCESS(urQueueRelease(queues[i]));
    }
    for (size_t i = 0; i < contexts.size(); ++i) {
      EXPECT_SUCCESS(urContextRelease(contexts[i]));
    }
    UUR_RETURN_ON_FATAL_FAILURE(urDeviceTest::TearDown());
  }

  const size_t maxNumContexts = 5;
  std::vector<ur_context_handle_t> contexts;
  std::vector<ur_queue_handle_t> queues;
  std::vector<ur_mem_handle_t> src_buffer;
  std::vector<ur_mem_handle_t> dst_buffer;
  const size_t count = 1024;
  const size_t size = sizeof(uint32_t) * count;
  std::vector<std::vector<uint32_t>> input;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEventWaitTest);

TEST_P(urEventWaitTest, Success) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  ur_event_handle_t event1 = nullptr;
  ASSERT_SUCCESS(urEnqueueMemBufferCopy(queues[0], src_buffer[0], dst_buffer[0],
                                        0, 0, size, 0, nullptr, &event1));
  std::vector<uint32_t> output(count, 1);
  ur_event_handle_t event2 = nullptr;
  ASSERT_SUCCESS(urEnqueueMemBufferRead(queues[0], dst_buffer[0], false, 0,
                                        size, output.data(), 0, nullptr,
                                        &event2));
  std::vector<ur_event_handle_t> events{event1, event2};
  EXPECT_SUCCESS(urQueueFlush(queues[0]));
  ASSERT_SUCCESS(
      urEventWait(static_cast<uint32_t>(events.size()), events.data()));
  ASSERT_EQ(input[0], output);

  EXPECT_SUCCESS(urEventRelease(event1));
  EXPECT_SUCCESS(urEventRelease(event2));
}

using urEventWaitNegativeTest = uur::urQueueTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEventWaitNegativeTest);

TEST_P(urEventWaitNegativeTest, ZeroSize) {
  ur_event_handle_t event = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_VALUE, urEventWait(0, &event));
}

TEST_P(urEventWaitNegativeTest, InvalidNullPointerEventList) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urEventWait(1, nullptr));
}

TEST_P(urEventWaitTest, WaitWithMultipleContexts) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  for (size_t i = 0; i < maxNumContexts; i++) {
    ASSERT_SUCCESS(urEnqueueMemBufferCopy(queues[i], src_buffer[i],
                                          dst_buffer[i], 0, 0, size, 0, nullptr,
                                          nullptr));
  }

  std::vector<ur_event_handle_t> events;
  std::vector<std::vector<uint32_t>> output;
  for (size_t i = 0; i < maxNumContexts; i++) {
    output.emplace_back(count, 1);
    events.emplace_back();
    ASSERT_SUCCESS(urEnqueueMemBufferRead(queues[i], dst_buffer[i], false, 0,
                                          size, output[i].data(), 0, nullptr,
                                          &events.back()));
  }

  ASSERT_SUCCESS(
      urEventWait(static_cast<uint32_t>(events.size()), events.data()));

  for (size_t i = 0; i < maxNumContexts; i++) {
    ASSERT_EQ(input[i], output[i]);
  }

  for (auto &event : events) {
    EXPECT_SUCCESS(urEventRelease(event));
  }
}

TEST_P(urEventWaitTest, WithCrossContextDependencies) {
  // OpenCL: https://github.com/intel/llvm/issues/18765
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{}, uur::OpenCL{});

  std::vector<uint32_t> output(count, 1);

  std::vector<ur_event_handle_t> events;
  for (size_t i = 0; i < maxNumContexts - 1; i++) {
    auto waitEvent = events.size() ? &events.back() : nullptr;
    ur_event_handle_t event = nullptr;
    ASSERT_SUCCESS(
        urEnqueueMemBufferCopy(queues[i], src_buffer[i], src_buffer[i + 1], 0,
                               0, size, waitEvent ? 1 : 0, waitEvent, &event));
    events.push_back(event);
  }

  ur_event_handle_t event1 = nullptr;
  ASSERT_SUCCESS(urEnqueueMemBufferCopy(queues.back(), src_buffer.back(),
                                        dst_buffer.back(), 0, 0, size, 1,
                                        &events.back(), &event1));

  ur_event_handle_t event2 = nullptr;
  ASSERT_SUCCESS(urEnqueueMemBufferRead(queues.back(), dst_buffer.back(), false,
                                        0, size, output.data(), 0, nullptr,
                                        &event2));

  events.push_back(event1);
  events.push_back(event2);

  ASSERT_SUCCESS(
      urEventWait(static_cast<uint32_t>(events.size()), events.data()));
  ASSERT_EQ(input.front(), output);

  EXPECT_SUCCESS(urEventRelease(event1));
  EXPECT_SUCCESS(urEventRelease(event2));
}
