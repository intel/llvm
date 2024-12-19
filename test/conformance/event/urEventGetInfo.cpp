// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urEventGetInfoTest = uur::event::urEventTest;

TEST_P(urEventGetInfoTest, SuccessCommandQueue) {
    ur_event_info_t info_type = UR_EVENT_INFO_COMMAND_QUEUE;
    size_t size = 0;

    ASSERT_SUCCESS(urEventGetInfo(event, info_type, 0, nullptr, &size));
    ASSERT_NE(size, 0);
    ASSERT_EQ(size, sizeof(ur_queue_handle_t));

    ur_queue_handle_t returned_queue;
    ASSERT_SUCCESS(
        urEventGetInfo(event, info_type, size, &returned_queue, nullptr));

    ASSERT_EQ(queue, returned_queue);
}

TEST_P(urEventGetInfoTest, SuccessContext) {
    ur_event_info_t info_type = UR_EVENT_INFO_CONTEXT;
    size_t size = 0;

    ASSERT_SUCCESS(urEventGetInfo(event, info_type, 0, nullptr, &size));
    ASSERT_NE(size, 0);
    ASSERT_EQ(size, sizeof(ur_context_handle_t));

    ur_context_handle_t returned_context;
    ASSERT_SUCCESS(
        urEventGetInfo(event, info_type, size, &returned_context, nullptr));

    ASSERT_EQ(context, returned_context);
}

TEST_P(urEventGetInfoTest, SuccessCommandType) {
    ur_event_info_t info_type = UR_EVENT_INFO_COMMAND_TYPE;
    size_t size = 0;

    ASSERT_SUCCESS(urEventGetInfo(event, info_type, 0, nullptr, &size));
    ASSERT_NE(size, 0);
    ASSERT_EQ(size, sizeof(ur_command_t));

    ur_command_t returned_command_type;
    ASSERT_SUCCESS(urEventGetInfo(event, info_type, size,
                                  &returned_command_type, nullptr));

    ASSERT_EQ(UR_COMMAND_MEM_BUFFER_WRITE, returned_command_type);
}

TEST_P(urEventGetInfoTest, SuccessCommandExecutionStatus) {
    ur_event_info_t info_type = UR_EVENT_INFO_COMMAND_EXECUTION_STATUS;
    size_t size = 0;

    ASSERT_SUCCESS(urEventGetInfo(event, info_type, 0, nullptr, &size));
    ASSERT_NE(size, 0);
    ASSERT_EQ(size, sizeof(ur_event_status_t));

    ur_event_status_t returned_status;
    ASSERT_SUCCESS(
        urEventGetInfo(event, info_type, size, &returned_status, nullptr));

    ASSERT_EQ(UR_EVENT_STATUS_COMPLETE, returned_status);
}

TEST_P(urEventGetInfoTest, SuccessReferenceCount) {
    ur_event_info_t info_type = UR_EVENT_INFO_REFERENCE_COUNT;
    size_t size = 0;

    ASSERT_SUCCESS(urEventGetInfo(event, info_type, 0, nullptr, &size));
    ASSERT_NE(size, 0);
    ASSERT_EQ(size, sizeof(uint32_t));

    uint32_t returned_reference_count;
    ASSERT_SUCCESS(urEventGetInfo(event, info_type, size,
                                  &returned_reference_count, nullptr));

    ASSERT_GT(returned_reference_count, 0U);
}

TEST_P(urEventGetInfoTest, InvalidNullHandle) {
    ur_event_info_t info_type = UR_EVENT_INFO_COMMAND_QUEUE;
    size_t size;
    ASSERT_SUCCESS(urEventGetInfo(event, info_type, 0, nullptr, &size));
    ASSERT_NE(size, 0);
    std::vector<uint8_t> data(size);

    /* Invalid hEvent */
    ASSERT_EQ_RESULT(
        urEventGetInfo(nullptr, UR_EVENT_INFO_COMMAND_QUEUE, 0, nullptr, &size),
        UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEventGetInfoTest, InvalidEnumeration) {
    size_t size;
    ASSERT_EQ_RESULT(
        urEventGetInfo(event, UR_EVENT_INFO_FORCE_UINT32, 0, nullptr, &size),
        UR_RESULT_ERROR_INVALID_ENUMERATION);
}

TEST_P(urEventGetInfoTest, InvalidSizePropSize) {
    ur_event_info_t info_type = UR_EVENT_INFO_COMMAND_QUEUE;
    size_t size = 0;
    ASSERT_SUCCESS(urEventGetInfo(event, info_type, 0, nullptr, &size));
    ASSERT_NE(size, 0);
    std::vector<uint8_t> data(size);

    /* Invalid propSize */
    ASSERT_EQ_RESULT(urEventGetInfo(event, UR_EVENT_INFO_COMMAND_QUEUE, 0,
                                    data.data(), nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urEventGetInfoTest, InvalidSizePropSizeSmall) {
    ur_queue_handle_t q;
    ASSERT_EQ_RESULT(urEventGetInfo(event, UR_EVENT_INFO_COMMAND_QUEUE,
                                    sizeof(q) - 1, &q, nullptr),
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
