// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.hpp"
#include "ur_api.h"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

TEST_F(LoaderHandleTest, Success) {
    ur_platform_handle_t query_platform;
    size_t retsize;
    ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_PLATFORM,
                                   sizeof(intptr_t), &query_platform,
                                   &retsize));
    ASSERT_EQ(query_platform, platform);
}

TEST_F(LoaderHandleTest, SuccessArray) {
    ur_platform_handle_t query_platform[2] = {(ur_platform_handle_t)0xCAFE,
                                              (ur_platform_handle_t)0xBEEF};
    ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_PLATFORM,
                                   sizeof(query_platform), &query_platform,
                                   NULL));
    ASSERT_EQ(query_platform[0], platform);
    ASSERT_EQ(query_platform[1], (ur_platform_handle_t)0xBEEF);
}

TEST_F(LoaderHandleTest, SuccessArraySizeRet) {
    ur_platform_handle_t query_platform[2] = {(ur_platform_handle_t)0xCAFE,
                                              (ur_platform_handle_t)0xBEEF};
    size_t sizeret;
    ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_PLATFORM,
                                   sizeof(query_platform), &query_platform,
                                   &sizeret));
    ASSERT_EQ(sizeret, sizeof(intptr_t));
    ASSERT_EQ(query_platform[0], platform);
    ASSERT_EQ(query_platform[1], (ur_platform_handle_t)0xBEEF);
}
