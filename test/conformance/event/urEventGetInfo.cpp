// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include "uur/known_failure.h"

using urEventGetInfoTest = uur::event::urEventTest;

TEST_P(urEventGetInfoTest, SuccessCommandQueue) {
    ur_event_info_t property_name = UR_EVENT_INFO_COMMAND_QUEUE;
    size_t property_size = 0;

    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urEventGetInfo(event, property_name, 0, nullptr, &property_size),
        property_name);
    ASSERT_EQ(property_size, sizeof(ur_queue_handle_t));

    ur_queue_handle_t returned_queue = nullptr;
    ASSERT_SUCCESS(urEventGetInfo(event, property_name, property_size,
                                  &returned_queue, nullptr));

    ASSERT_EQ(queue, returned_queue);
}

TEST_P(urEventGetInfoTest, SuccessContext) {
    ur_event_info_t property_name = UR_EVENT_INFO_CONTEXT;
    size_t property_size = 0;

    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urEventGetInfo(event, property_name, 0, nullptr, &property_size),
        property_name);
    ASSERT_EQ(property_size, sizeof(ur_context_handle_t));

    ur_context_handle_t returned_context = nullptr;
    ASSERT_SUCCESS(urEventGetInfo(event, property_name, property_size,
                                  &returned_context, nullptr));

    ASSERT_EQ(context, returned_context);
}

TEST_P(urEventGetInfoTest, SuccessCommandType) {
    UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

    ur_event_info_t property_name = UR_EVENT_INFO_COMMAND_TYPE;
    size_t property_size = 0;

    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urEventGetInfo(event, property_name, 0, nullptr, &property_size),
        property_name);
    ASSERT_EQ(property_size, sizeof(ur_command_t));

    ur_command_t returned_command_type = UR_COMMAND_FORCE_UINT32;
    ASSERT_SUCCESS(urEventGetInfo(event, property_name, property_size,
                                  &returned_command_type, nullptr));

    ASSERT_EQ(UR_COMMAND_MEM_BUFFER_WRITE, returned_command_type);
}

TEST_P(urEventGetInfoTest, SuccessCommandExecutionStatus) {
    ur_event_info_t property_name = UR_EVENT_INFO_COMMAND_EXECUTION_STATUS;
    size_t property_size = 0;

    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urEventGetInfo(event, property_name, 0, nullptr, &property_size),
        property_name);
    ASSERT_EQ(property_size, sizeof(ur_event_status_t));

    ur_event_status_t returned_status = UR_EVENT_STATUS_FORCE_UINT32;
    ASSERT_SUCCESS(urEventGetInfo(event, property_name, property_size,
                                  &returned_status, nullptr));

    ASSERT_EQ(UR_EVENT_STATUS_COMPLETE, returned_status);
}

TEST_P(urEventGetInfoTest, SuccessReferenceCount) {
    ur_event_info_t property_name = UR_EVENT_INFO_REFERENCE_COUNT;
    size_t property_size = 0;

    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urEventGetInfo(event, property_name, 0, nullptr, &property_size),
        property_name);
    ASSERT_EQ(property_size, sizeof(uint32_t));

    uint32_t returned_reference_count = 0;
    ASSERT_SUCCESS(urEventGetInfo(event, property_name, property_size,
                                  &returned_reference_count, nullptr));

    ASSERT_GT(returned_reference_count, 0U);
}

TEST_P(urEventGetInfoTest, InvalidNullHandle) {
    ur_event_info_t property_name = UR_EVENT_INFO_COMMAND_QUEUE;
    size_t property_size;

    ASSERT_SUCCESS(
        urEventGetInfo(event, property_name, 0, nullptr, &property_size));
    ASSERT_NE(property_size, 0);
    std::vector<uint8_t> data(property_size);

    /* Invalid hEvent */
    ASSERT_EQ_RESULT(urEventGetInfo(nullptr, UR_EVENT_INFO_COMMAND_QUEUE, 0,
                                    nullptr, &property_size),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEventGetInfoTest, InvalidEnumeration) {
    size_t property_size = 0;

    ASSERT_EQ_RESULT(urEventGetInfo(event, UR_EVENT_INFO_FORCE_UINT32, 0,
                                    nullptr, &property_size),
                     UR_RESULT_ERROR_INVALID_ENUMERATION);
}

TEST_P(urEventGetInfoTest, InvalidSizePropSize) {
    ur_event_info_t property_name = UR_EVENT_INFO_COMMAND_QUEUE;
    size_t property_size = 0;

    ASSERT_SUCCESS(
        urEventGetInfo(event, property_name, 0, nullptr, &property_size));
    ASSERT_NE(property_size, 0);
    std::vector<uint8_t> data(property_size);

    /* Invalid propSize */
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
                                    sizeof(ur_queue_handle_t), nullptr,
                                    nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urEventGetInfoTest, InvalidNullPointerPropSizeRet) {
    ASSERT_EQ_RESULT(
        urEventGetInfo(event, UR_EVENT_INFO_COMMAND_QUEUE, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventGetInfoTest);
