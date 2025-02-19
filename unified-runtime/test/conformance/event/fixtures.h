// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_CONFORMANCE_EVENT_FIXTURES_H_INCLUDED
#define UR_CONFORMANCE_EVENT_FIXTURES_H_INCLUDED

#include <uur/fixtures.h>

namespace uur {
namespace event {

/**
 * Test fixture that sets up an event with the following properties:
 * - Type: UR_COMMAND_MEM_BUFFER_WRITE
 * - Execution Status: UR_EVENT_STATUS_COMPLETE
 * - Reference Count: 1
 */
template <class T>
struct urEventTestWithParam : uur::urProfilingQueueTestWithParam<T> {

  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urProfilingQueueTestWithParam<T>::SetUp());
    ASSERT_SUCCESS(urMemBufferCreate(this->context, UR_MEM_FLAG_WRITE_ONLY,
                                     size, nullptr, &buffer));

    input.assign(count, 42);
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(this->queue, buffer, false, 0, size,
                                           input.data(), 0, nullptr, &event));
    ASSERT_SUCCESS(urEventWait(1, &event));
  }

  void TearDown() override {
    if (buffer) {
      EXPECT_SUCCESS(urMemRelease(buffer));
    }
    if (event) {
      EXPECT_SUCCESS(urEventRelease(event));
    }
    uur::urProfilingQueueTestWithParam<T>::TearDown();
  }

  const size_t count = 1024;
  const size_t size = sizeof(uint32_t) * count;
  ur_mem_handle_t buffer = nullptr;
  ur_event_handle_t event = nullptr;
  std::vector<uint32_t> input;
};

/**
 * Test fixture that is intended to be used when testing reference count APIs
 * (i.e. urEventRelease and urEventRetain). Does not handle destruction of the
 * event.
 */
struct urEventReferenceTest : uur::urProfilingQueueTest {

  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urProfilingQueueTest::SetUp());
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_WRITE_ONLY, size,
                                     nullptr, &buffer));

    input.assign(count, 42);
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, false, 0, size,
                                           input.data(), 0, nullptr, &event));
    ASSERT_SUCCESS(urEventWait(1, &event));
  }

  void TearDown() override {
    if (buffer) {
      EXPECT_SUCCESS(urMemRelease(buffer));
    }
    urProfilingQueueTest::TearDown();
  }

  const size_t count = 1024;
  const size_t size = sizeof(uint32_t) * count;
  ur_mem_handle_t buffer = nullptr;
  ur_event_handle_t event = nullptr;
  std::vector<uint32_t> input;
};

struct urEventTest : urEventReferenceTest {

  void TearDown() override {
    if (event) {
      EXPECT_SUCCESS(urEventRelease(event));
    }
    urEventReferenceTest::TearDown();
  }
};
} // namespace event
} // namespace uur

#endif // UR_CONFORMANCE_EVENT_FIXTURES_H_INCLUDED
