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
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{}, uur::NativeCPU{});

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
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{}, uur::NativeCPU{});

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
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{}, uur::LevelZero{},
                       uur::NativeCPU{});

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
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{}, uur::LevelZero{},
                       uur::LevelZeroV2{}, uur::NativeCPU{});

  uint8_t size = 8;

  uint64_t queued_value = 0;
  ASSERT_SUCCESS(urEventGetProfilingInfo(
      event, UR_PROFILING_INFO_COMMAND_QUEUED, size, &queued_value, nullptr));
  ASSERT_NE(queued_value, 0);

  uint64_t submit_value = 0;
  ASSERT_SUCCESS(urEventGetProfilingInfo(
      event, UR_PROFILING_INFO_COMMAND_SUBMIT, size, &submit_value, nullptr));
  ASSERT_NE(submit_value, 0);

  uint64_t start_value = 0;
  ASSERT_SUCCESS(urEventGetProfilingInfo(event, UR_PROFILING_INFO_COMMAND_START,
                                         size, &start_value, nullptr));
  ASSERT_NE(start_value, 0);

  uint64_t end_value = 0;
  ASSERT_SUCCESS(urEventGetProfilingInfo(event, UR_PROFILING_INFO_COMMAND_END,
                                         size, &end_value, nullptr));
  ASSERT_NE(end_value, 0);

  uint64_t complete_value = 0;
  ASSERT_SUCCESS(urEventGetProfilingInfo(event,
                                         UR_PROFILING_INFO_COMMAND_COMPLETE,
                                         size, &complete_value, nullptr));
  ASSERT_NE(complete_value, 0);

  ASSERT_LE(queued_value, submit_value);
  ASSERT_LT(submit_value, start_value);
  ASSERT_LT(start_value, end_value);
  ASSERT_LE(end_value, complete_value);
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
