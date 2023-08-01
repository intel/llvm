// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urPlatformGetApiVersionTest = uur::platform::urPlatformTest;

TEST_F(urPlatformGetApiVersionTest, Success) {
    ur_api_version_t version;
    ASSERT_EQ_RESULT(UR_RESULT_SUCCESS,
                     urPlatformGetApiVersion(platform, &version));
    ASSERT_GE(UR_API_VERSION_CURRENT, version);
}

TEST_F(urPlatformGetApiVersionTest, InvalidPlatform) {
    ur_api_version_t version;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urPlatformGetApiVersion(nullptr, &version));
}

TEST_F(urPlatformGetApiVersionTest, InvalidVersionPtr) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urPlatformGetApiVersion(platform, nullptr));
}
