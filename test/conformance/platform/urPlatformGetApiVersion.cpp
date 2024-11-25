// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urPlatformGetApiVersionTest = uur::urPlatformTest;
UUR_INSTANTIATE_PLATFORM_TEST_SUITE_P(urPlatformGetApiVersionTest);

TEST_P(urPlatformGetApiVersionTest, Success) {
    ur_api_version_t version;
    ASSERT_EQ_RESULT(UR_RESULT_SUCCESS,
                     urPlatformGetApiVersion(platform, &version));
    ASSERT_GE(UR_API_VERSION_CURRENT, version);
}

TEST_P(urPlatformGetApiVersionTest, InvalidPlatform) {
    ur_api_version_t version;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urPlatformGetApiVersion(nullptr, &version));
}

TEST_P(urPlatformGetApiVersionTest, InvalidVersionPtr) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urPlatformGetApiVersion(platform, nullptr));
}
