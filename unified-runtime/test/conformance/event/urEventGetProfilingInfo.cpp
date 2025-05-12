// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include "uur/known_failure.h"

using urEventGetProfilingInfoTest = uur::event::urEventTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEventGetProfilingInfoTest);

TEST_P(urEventGetProfilingInfoTest, SuccessCommandQueued) {
  const ur_profiling_info_t property_name = UR_PROFILING_INFO_COMMAND_QUEUED;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urEventGetProfilingInfo(event, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint64_t));

  uint64_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urEventGetProfilingInfo(event, property_name,
                                                     property_size,
                                                     &property_value, nullptr),
                             property_value);
}

TEST_P(urEventGetProfilingInfoTest, SuccessCommandSubmit) {
  const ur_profiling_info_t property_name = UR_PROFILING_INFO_COMMAND_SUBMIT;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urEventGetProfilingInfo(event, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint64_t));

  uint64_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urEventGetProfilingInfo(event, property_name,
                                                     property_size,
                                                     &property_value, nullptr),
                             property_value);
}

TEST_P(urEventGetProfilingInfoTest, SuccessCommandStart) {
  const ur_profiling_info_t property_name = UR_PROFILING_INFO_COMMAND_START;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urEventGetProfilingInfo(event, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint64_t));

  uint64_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urEventGetProfilingInfo(event, property_name,
                                                     property_size,
                                                     &property_value, nullptr),
                             property_value);
}

TEST_P(urEventGetProfilingInfoTest, SuccessCommandEnd) {
  const ur_profiling_info_t property_name = UR_PROFILING_INFO_COMMAND_END;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urEventGetProfilingInfo(event, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint64_t));

  uint64_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urEventGetProfilingInfo(event, property_name,
                                                     property_size,
                                                     &property_value, nullptr),
                             property_value);
}

TEST_P(urEventGetProfilingInfoTest, SuccessCommandComplete) {
  const ur_profiling_info_t property_name = UR_PROFILING_INFO_COMMAND_COMPLETE;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urEventGetProfilingInfo(event, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint64_t));

  uint64_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urEventGetProfilingInfo(event, property_name,
                                                     property_size,
                                                     &property_value, nullptr),
                             property_value);
}

TEST_P(urEventGetProfilingInfoTest, Success) {
  // AMD devices may report a "start" time before the "submit" time
  UUR_KNOWN_FAILURE_ON(uur::HIP{});

  // If a and b are supported, asserts that a <= b
  auto test_timing = [=](ur_profiling_info_t a, ur_profiling_info_t b) {
    std::stringstream trace{"Profiling Info: "};
    trace << a << " <= " << b;
    SCOPED_TRACE(trace.str());
    uint64_t a_time;
    auto result =
        urEventGetProfilingInfo(event, a, sizeof(a_time), &a_time, nullptr);
    if (result == UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION) {
      return;
    }
    ASSERT_SUCCESS(result);

    uint64_t b_time;
    result =
        urEventGetProfilingInfo(event, b, sizeof(b_time), &b_time, nullptr);
    if (result == UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION) {
      return;
    }
    ASSERT_SUCCESS(result);

    // Note: This assumes that the counter doesn't overflow
    ASSERT_LE(a_time, b_time);
  };

  test_timing(UR_PROFILING_INFO_COMMAND_QUEUED,
              UR_PROFILING_INFO_COMMAND_SUBMIT);
  test_timing(UR_PROFILING_INFO_COMMAND_SUBMIT,
              UR_PROFILING_INFO_COMMAND_START);
  test_timing(UR_PROFILING_INFO_COMMAND_START, UR_PROFILING_INFO_COMMAND_END);
  test_timing(UR_PROFILING_INFO_COMMAND_END,
              UR_PROFILING_INFO_COMMAND_COMPLETE);
}

TEST_P(urEventGetProfilingInfoTest, ReleaseEventAfterQueueRelease) {
  void *ptr;
  ASSERT_SUCCESS(
      urUSMSharedAlloc(context, device, nullptr, nullptr, 1024 * 1024, &ptr));

  // Enqueue an operation to keep the device busy
  uint8_t pattern = 0xFF;
  ur_event_handle_t event1;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(uint8_t), &pattern,
                                  1024 * 1024, 0, nullptr, &event1));

  ASSERT_SUCCESS(urQueueRelease(queue));
  queue = nullptr;

  uint64_t queuedTime = 0;
  auto ret = urEventGetProfilingInfo(event1, UR_PROFILING_INFO_COMMAND_QUEUED,
                                     sizeof(uint64_t), &queuedTime, nullptr);
  ASSERT_TRUE(ret == UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION ||
              ret == UR_RESULT_SUCCESS);

  ASSERT_SUCCESS(urEventRelease(event1));
  ASSERT_SUCCESS(urUSMFree(context, ptr));
}

TEST_P(urEventGetProfilingInfoTest, InvalidNullHandle) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  const ur_profiling_info_t property_name = UR_PROFILING_INFO_COMMAND_QUEUED;
  size_t property_size;
  ASSERT_SUCCESS(urEventGetProfilingInfo(event, property_name, 0, nullptr,
                                         &property_size));
  ASSERT_NE(property_size, 0);

  ASSERT_EQ_RESULT(urEventGetProfilingInfo(nullptr, property_name, 0, nullptr,
                                           &property_size),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEventGetProfilingInfoTest, InvalidEnumeration) {
  size_t property_size;
  ASSERT_EQ_RESULT(urEventGetProfilingInfo(event,
                                           UR_PROFILING_INFO_FORCE_UINT32, 0,
                                           nullptr, &property_size),
                   UR_RESULT_ERROR_INVALID_ENUMERATION);
}

TEST_P(urEventGetProfilingInfoTest, InvalidValue) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  const ur_profiling_info_t property_name = UR_PROFILING_INFO_COMMAND_QUEUED;
  size_t property_size = 0;
  ASSERT_SUCCESS(urEventGetProfilingInfo(event, property_name, 0, nullptr,
                                         &property_size));
  ASSERT_NE(property_size, 0);

  uint64_t property_value = 0;
  ASSERT_EQ_RESULT(urEventGetProfilingInfo(event, property_name, 0,
                                           &property_value, nullptr),
                   UR_RESULT_ERROR_INVALID_VALUE);
}

struct urEventGetProfilingInfoForWaitWithBarrier : uur::urProfilingQueueTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urProfilingQueueTest::SetUp());
    ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_WRITE_ONLY, size,
                                     nullptr, &buffer));

    input.assign(count, 42);
    ur_event_handle_t membuf_event = nullptr;
    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queue, buffer, false, 0, size,
                                           input.data(), 0, nullptr,
                                           &membuf_event));

    ASSERT_SUCCESS(
        urEnqueueEventsWaitWithBarrier(queue, 1, &membuf_event, &event));
    ASSERT_SUCCESS(urQueueFinish(queue));
  }

  void TearDown() override {
    UUR_RETURN_ON_FATAL_FAILURE(urProfilingQueueTest::TearDown());
  }

  const size_t count = 1024;
  const size_t size = sizeof(uint32_t) * count;
  ur_mem_handle_t buffer = nullptr;
  ur_event_handle_t event = nullptr;
  std::vector<uint32_t> input;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEventGetProfilingInfoForWaitWithBarrier);

TEST_P(urEventGetProfilingInfoForWaitWithBarrier, Success) {
  uint64_t submit_value = 0;
  ASSERT_SUCCESS(urEventGetProfilingInfo(event, UR_PROFILING_INFO_COMMAND_START,
                                         size, &submit_value, nullptr));
  ASSERT_NE(submit_value, 0);

  uint64_t complete_value = 0;
  ASSERT_SUCCESS(urEventGetProfilingInfo(event, UR_PROFILING_INFO_COMMAND_END,
                                         size, &complete_value, nullptr));
  ASSERT_NE(complete_value, 0);

  ASSERT_GE(complete_value, submit_value);
}

struct urEventGetProfilingInfoInvalidQueue : uur::urQueueTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());
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
    UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::TearDown());
  };

  const size_t count = 1024;
  const size_t size = sizeof(uint32_t) * count;
  ur_mem_handle_t buffer = nullptr;
  ur_event_handle_t event = nullptr;
  std::vector<uint32_t> input;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEventGetProfilingInfoInvalidQueue);

TEST_P(urEventGetProfilingInfoInvalidQueue, ProfilingInfoNotAvailable) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  const ur_profiling_info_t property_name = UR_PROFILING_INFO_COMMAND_QUEUED;
  size_t property_size;
  ASSERT_EQ_RESULT(
      urEventGetProfilingInfo(event, property_name, 0, nullptr, &property_size),
      UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE);
}
