// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include "uur/known_failure.h"

using urEventGetInfoTest = uur::event::urEventTest;

TEST_P(urEventGetInfoTest, SuccessCommandQueue) {
  const ur_event_info_t property_name = UR_EVENT_INFO_COMMAND_QUEUE;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urEventGetInfo(event, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_queue_handle_t));

  ur_queue_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urEventGetInfo(event, property_name, property_size,
                                &property_value, nullptr));

  ASSERT_EQ(queue, property_value);
}

TEST_P(urEventGetInfoTest, SuccessRoundtripContext) {
  // Segfaults
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  const ur_event_info_t property_name = UR_EVENT_INFO_CONTEXT;
  size_t property_size = sizeof(ur_context_handle_t);

  ur_native_handle_t native_event;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventGetNativeHandle(event, &native_event));

  ur_event_handle_t from_native_event;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urEventCreateWithNativeHandle(
      native_event, context, nullptr, &from_native_event));

  ur_context_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urEventGetInfo(from_native_event, property_name, property_size,
                                &property_value, nullptr));

  ASSERT_EQ(property_value, context);
}

TEST_P(urEventGetInfoTest, SuccessRoundtripCommandQueue) {
  UUR_KNOWN_FAILURE_ON(uur::HIP{}, uur::CUDA{});
  // Segfaults
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  const ur_event_info_t property_name = UR_EVENT_INFO_COMMAND_QUEUE;
  size_t property_size = sizeof(ur_queue_handle_t);

  ur_native_handle_t native_event;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventGetNativeHandle(event, &native_event));

  ur_event_handle_t from_native_event;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urEventCreateWithNativeHandle(
      native_event, context, nullptr, &from_native_event));

  ur_queue_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urEventGetInfo(from_native_event, property_name, property_size,
                                &property_value, nullptr));

  // We can't assume that the two queue handles are equal (since creating the
  // link to the UR structures has been severed by going through native handle,
  // so just check the underlying native pointers
  ur_native_handle_t original_queue;
  ur_native_handle_t new_queue;
  ASSERT_SUCCESS(urQueueGetNativeHandle(queue, nullptr, &original_queue));
  ASSERT_SUCCESS(urQueueGetNativeHandle(property_value, nullptr, &new_queue));
  ASSERT_EQ(original_queue, new_queue);
}

TEST_P(urEventGetInfoTest, SuccessContext) {
  const ur_event_info_t property_name = UR_EVENT_INFO_CONTEXT;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urEventGetInfo(event, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_context_handle_t));

  ur_context_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urEventGetInfo(event, property_name, property_size,
                                &property_value, nullptr));

  ASSERT_EQ(context, property_value);
}

TEST_P(urEventGetInfoTest, SuccessCommandType) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  const ur_event_info_t property_name = UR_EVENT_INFO_COMMAND_TYPE;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urEventGetInfo(event, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_command_t));

  ur_command_t property_value = UR_COMMAND_FORCE_UINT32;
  ASSERT_SUCCESS(urEventGetInfo(event, property_name, property_size,
                                &property_value, nullptr));

  ASSERT_EQ(UR_COMMAND_MEM_BUFFER_WRITE, property_value);
}

TEST_P(urEventGetInfoTest, SuccessCommandExecutionStatus) {
  const ur_event_info_t property_name = UR_EVENT_INFO_COMMAND_EXECUTION_STATUS;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urEventGetInfo(event, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_event_status_t));

  ur_event_status_t property_value = UR_EVENT_STATUS_FORCE_UINT32;
  ASSERT_SUCCESS(urEventGetInfo(event, property_name, property_size,
                                &property_value, nullptr));

  ASSERT_EQ(UR_EVENT_STATUS_COMPLETE, property_value);
}

TEST_P(urEventGetInfoTest, SuccessReferenceCount) {
  const ur_event_info_t property_name = UR_EVENT_INFO_REFERENCE_COUNT;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urEventGetInfo(event, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urEventGetInfo(event, property_name, property_size,
                                            &property_value, nullptr),
                             property_value);

  ASSERT_GT(property_value, 0U);
}

TEST_P(urEventGetInfoTest, InvalidNullHandle) {
  const ur_event_info_t property_name = UR_EVENT_INFO_COMMAND_QUEUE;
  size_t property_size;

  ASSERT_SUCCESS(
      urEventGetInfo(event, property_name, 0, nullptr, &property_size));
  ASSERT_NE(property_size, 0);

  ASSERT_EQ_RESULT(urEventGetInfo(nullptr, UR_EVENT_INFO_COMMAND_QUEUE, 0,
                                  nullptr, &property_size),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEventGetInfoTest, InvalidEnumeration) {
  size_t property_size = 0;

  ASSERT_EQ_RESULT(urEventGetInfo(event, UR_EVENT_INFO_FORCE_UINT32, 0, nullptr,
                                  &property_size),
                   UR_RESULT_ERROR_INVALID_ENUMERATION);
}

TEST_P(urEventGetInfoTest, InvalidSizePropSize) {
  const ur_event_info_t property_name = UR_EVENT_INFO_COMMAND_QUEUE;
  size_t property_size = 0;

  ASSERT_SUCCESS(
      urEventGetInfo(event, property_name, 0, nullptr, &property_size));
  ASSERT_NE(property_size, 0);
  std::vector<uint8_t> data(property_size);

  ASSERT_EQ_RESULT(urEventGetInfo(event, UR_EVENT_INFO_COMMAND_QUEUE, 0,
                                  data.data(), nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urEventGetInfoTest, InvalidSizePropSizeSmall) {
  ur_queue_handle_t queue = nullptr;

  ASSERT_EQ_RESULT(urEventGetInfo(event, UR_EVENT_INFO_COMMAND_QUEUE,
                                  sizeof(queue) - 1, &queue, nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urEventGetInfoTest, InvalidNullPointerPropValue) {
  ASSERT_EQ_RESULT(urEventGetInfo(event, UR_EVENT_INFO_COMMAND_QUEUE,
                                  sizeof(ur_queue_handle_t), nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urEventGetInfoTest, InvalidNullPointerPropSizeRet) {
  ASSERT_EQ_RESULT(
      urEventGetInfo(event, UR_EVENT_INFO_COMMAND_QUEUE, 0, nullptr, nullptr),
      UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEventGetInfoTest);
