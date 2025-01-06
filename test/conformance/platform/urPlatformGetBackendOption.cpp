// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "fixtures.h"

using urPlatformGetBackendOptionTest =
    uur::platform::urPlatformTestWithParam<std::string>;

UUR_PLATFORM_TEST_SUITE_P(urPlatformGetBackendOptionTest,
                          ::testing::Values("-O0", "-O1", "-O2", "-O3"),
                          std::string);

TEST_P(urPlatformGetBackendOptionTest, Success) {
    const char *platformOption = nullptr;
    ASSERT_SUCCESS(urPlatformGetBackendOption(platform, getParam().c_str(),
                                              &platformOption));
    ASSERT_NE(platformOption, nullptr);
}

using urPlatformGetBackendOptionNegativeTest = uur::urPlatformTest;
UUR_INSTANTIATE_PLATFORM_TEST_SUITE_P(urPlatformGetBackendOptionNegativeTest);

TEST_P(urPlatformGetBackendOptionNegativeTest, InvalidNullHandle) {
    const char *platformOption = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urPlatformGetBackendOption(nullptr, "-O0", &platformOption));
}

TEST_P(urPlatformGetBackendOptionNegativeTest,
       InvalidNullPointerFrontendOption) {
    const char *platformOption = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urPlatformGetBackendOption(platform, nullptr, &platformOption));
}

TEST_P(urPlatformGetBackendOptionNegativeTest,
       InvalidNullPointerPlatformOption) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urPlatformGetBackendOption(platform, "-O0", nullptr));
}

TEST_P(urPlatformGetBackendOptionNegativeTest, InvalidValueFrontendOption) {
    const char *platformOption = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_VALUE,
        urPlatformGetBackendOption(platform, "-invalid-opt", &platformOption));
}
