// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include "fixtures.h"

struct urPlatfromGetBackendOptionTestWithParam
    : uur::platform::urPlatformTest,
      ::testing::WithParamInterface<std::string> {};

INSTANTIATE_TEST_SUITE_P(, urPlatfromGetBackendOptionTestWithParam,
                         ::testing::Values("-O0", "-O1", "-O2", "-O3"),
                         [](const ::testing::TestParamInfo<std::string> &info) {
                             return uur::GTestSanitizeString(info.param);
                         });

TEST_P(urPlatfromGetBackendOptionTestWithParam, Success) {
    const char *adapterOption = nullptr;
    ASSERT_SUCCESS(urPlatformGetBackendOption(platform, GetParam().c_str(),
                                              &adapterOption));
    ASSERT_NE(adapterOption, nullptr);
}

using urPlatfromGetBackendOptionTest = uur::platform::urPlatformTest;

TEST_F(urPlatfromGetBackendOptionTest, InvalidNullHandle) {
    const char *adapterOption = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urPlatformGetBackendOption(nullptr, "-O0", &adapterOption));
}

TEST_F(urPlatfromGetBackendOptionTest, InvalidNullPointerFrontendOption) {
    const char *adapterOption = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urPlatformGetBackendOption(platform, nullptr, &adapterOption));
}

TEST_F(urPlatfromGetBackendOptionTest, InvalidNullPointerAdapterOption) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urPlatformGetBackendOption(platform, "-O0", nullptr));
}
