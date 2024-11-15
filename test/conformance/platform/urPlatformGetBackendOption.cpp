// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
    const char *platformOption = nullptr;
    ASSERT_SUCCESS(urPlatformGetBackendOption(platform, GetParam().c_str(),
                                              &platformOption));
    ASSERT_NE(platformOption, nullptr);
}

using urPlatfromGetBackendOptionTest = uur::platform::urPlatformTest;

TEST_F(urPlatfromGetBackendOptionTest, InvalidNullHandle) {
    const char *platformOption = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urPlatformGetBackendOption(nullptr, "-O0", &platformOption));
}

TEST_F(urPlatfromGetBackendOptionTest, InvalidNullPointerFrontendOption) {
    const char *platformOption = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urPlatformGetBackendOption(platform, nullptr, &platformOption));
}

TEST_F(urPlatfromGetBackendOptionTest, InvalidNullPointerPlatformOption) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urPlatformGetBackendOption(platform, "-O0", nullptr));
}

TEST_F(urPlatfromGetBackendOptionTest, InvalidValueFrontendOption) {
    const char *platformOption = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_VALUE,
        urPlatformGetBackendOption(platform, "-invalid-opt", &platformOption));
}
