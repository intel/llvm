// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include "fixtures.h"

using urEventGetProfilingInfoTest =
    uur::event::urEventTestWithParam<ur_profiling_info_t>;

TEST_P(urEventGetProfilingInfoTest, Success) {

    ur_profiling_info_t info_type = getParam();
    size_t size;
    ASSERT_SUCCESS(
        urEventGetProfilingInfo(event, info_type, 0, nullptr, &size));
    ASSERT_EQ(size, 8);

    std::vector<uint8_t> data(size);
    ASSERT_SUCCESS(
        urEventGetProfilingInfo(event, info_type, size, data.data(), nullptr));

    if (sizeof(size_t) == size) {
        auto returned_value = reinterpret_cast<size_t *>(data.data());
        ASSERT_NE(*returned_value, 0);
    }
}

UUR_TEST_SUITE_P(urEventGetProfilingInfoTest,
                 ::testing::Values(UR_PROFILING_INFO_COMMAND_QUEUED,
                                   UR_PROFILING_INFO_COMMAND_SUBMIT,
                                   UR_PROFILING_INFO_COMMAND_START,
                                   UR_PROFILING_INFO_COMMAND_END),
                 uur::deviceTestWithParamPrinter<ur_profiling_info_t>);

using urEventGetProfilingInfoNegativeTest = uur::event::urEventTest;

TEST_P(urEventGetProfilingInfoNegativeTest, InvalidNullHandle) {
    ur_profiling_info_t info_type = UR_PROFILING_INFO_COMMAND_QUEUED;
    size_t size;
    ASSERT_SUCCESS(
        urEventGetProfilingInfo(event, info_type, 0, nullptr, &size));
    ASSERT_NE(size, 0);
    std::vector<uint8_t> data(size);

    /* Invalid hEvent */
    ASSERT_EQ_RESULT(
        urEventGetProfilingInfo(nullptr, info_type, 0, nullptr, &size),
        UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEventGetProfilingInfoNegativeTest, InvalidEnumeration) {
    size_t size;
    ASSERT_EQ_RESULT(urEventGetProfilingInfo(event,
                                             UR_PROFILING_INFO_FORCE_UINT32, 0,
                                             nullptr, &size),
                     UR_RESULT_ERROR_INVALID_ENUMERATION);
}

TEST_P(urEventGetProfilingInfoNegativeTest, InvalidValue) {
    ur_profiling_info_t info_type = UR_PROFILING_INFO_COMMAND_QUEUED;
    size_t size;
    ASSERT_SUCCESS(
        urEventGetProfilingInfo(event, info_type, 0, nullptr, &size));
    ASSERT_NE(size, 0);
    std::vector<uint8_t> data(size);

    /* Invalid propValueSize */
    ASSERT_EQ_RESULT(
        urEventGetProfilingInfo(event, info_type, 0, data.data(), nullptr),
        UR_RESULT_ERROR_INVALID_VALUE);
}

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventGetProfilingInfoNegativeTest);
