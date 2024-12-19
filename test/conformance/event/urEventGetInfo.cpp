// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urEventGetInfoTest = uur::event::urEventTestWithParam<ur_event_info_t>;

TEST_P(urEventGetInfoTest, Success) {

    ur_event_info_t info_type = getParam();
    size_t size;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urEventGetInfo(event, info_type, 0, nullptr, &size), info_type);
    ASSERT_NE(size, 0);
    std::vector<uint8_t> data(size);
    ASSERT_SUCCESS(
        urEventGetInfo(event, info_type, size, data.data(), nullptr));

    switch (info_type) {
    case UR_EVENT_INFO_COMMAND_QUEUE: {
        ASSERT_EQ(sizeof(ur_queue_handle_t), size);
        auto returned_queue =
            reinterpret_cast<ur_queue_handle_t *>(data.data());
        ASSERT_EQ(queue, *returned_queue);
        break;
    }
    case UR_EVENT_INFO_CONTEXT: {
        ASSERT_EQ(sizeof(ur_context_handle_t), size);
        auto returned_context =
            reinterpret_cast<ur_context_handle_t *>(data.data());
        ASSERT_EQ(context, *returned_context);
        break;
    }
    case UR_EVENT_INFO_COMMAND_TYPE: {
        ASSERT_EQ(sizeof(ur_command_t), size);
        auto returned_command = reinterpret_cast<ur_command_t *>(data.data());
        ASSERT_EQ(UR_COMMAND_MEM_BUFFER_WRITE, *returned_command);
        break;
    }
    case UR_EVENT_INFO_COMMAND_EXECUTION_STATUS: {
        ASSERT_EQ(sizeof(ur_event_status_t), size);
        auto returned_status =
            reinterpret_cast<ur_event_status_t *>(data.data());
        ASSERT_EQ(UR_EVENT_STATUS_COMPLETE, *returned_status);
        break;
    }
    case UR_EVENT_INFO_REFERENCE_COUNT: {
        ASSERT_EQ(sizeof(uint32_t), size);
        auto returned_reference_count =
            reinterpret_cast<uint32_t *>(data.data());
        ASSERT_GT(*returned_reference_count, 0U);
        break;
    }
    default:
        FAIL() << "Invalid event info enumeration";
    }
}

UUR_TEST_SUITE_P(urEventGetInfoTest,
                 ::testing::Values(UR_EVENT_INFO_COMMAND_QUEUE,
                                   UR_EVENT_INFO_CONTEXT,
                                   UR_EVENT_INFO_COMMAND_TYPE,
                                   UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                                   UR_EVENT_INFO_REFERENCE_COUNT),
                 uur::deviceTestWithParamPrinter<ur_event_info_t>);

using urEventGetInfoNegativeTest = uur::event::urEventTest;

TEST_P(urEventGetInfoNegativeTest, InvalidNullHandle) {
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

TEST_P(urEventGetInfoNegativeTest, InvalidEnumeration) {
    size_t size;
    ASSERT_EQ_RESULT(
        urEventGetInfo(event, UR_EVENT_INFO_FORCE_UINT32, 0, nullptr, &size),
        UR_RESULT_ERROR_INVALID_ENUMERATION);
}

TEST_P(urEventGetInfoNegativeTest, InvalidSizePropSize) {
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

TEST_P(urEventGetInfoNegativeTest, InvalidSizePropSizeSmall) {
    ur_queue_handle_t q;
    ASSERT_EQ_RESULT(urEventGetInfo(event, UR_EVENT_INFO_COMMAND_QUEUE,
                                    sizeof(q) - 1, &q, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urEventGetInfoNegativeTest, InvalidNullPointerPropValue) {
    ASSERT_EQ_RESULT(urEventGetInfo(event, UR_EVENT_INFO_COMMAND_QUEUE,
                                    sizeof(ur_queue_handle_t), nullptr,
                                    nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urEventGetInfoNegativeTest, InvalidNullPointerPropSizeRet) {
    ASSERT_EQ_RESULT(
        urEventGetInfo(event, UR_EVENT_INFO_COMMAND_QUEUE, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventGetInfoNegativeTest);
