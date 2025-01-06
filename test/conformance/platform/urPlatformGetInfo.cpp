// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include "uur/environment.h"
#include "uur/fixtures.h"
#include <cstring>

using urPlatformGetInfoTest =
    uur::platform::urPlatformTestWithParam<ur_platform_info_t>;

UUR_PLATFORM_TEST_SUITE_P(
    urPlatformGetInfoTest,
    ::testing::Values(UR_PLATFORM_INFO_NAME, UR_PLATFORM_INFO_VENDOR_NAME,
                      UR_PLATFORM_INFO_VERSION, UR_PLATFORM_INFO_EXTENSIONS,
                      UR_PLATFORM_INFO_PROFILE, UR_PLATFORM_INFO_BACKEND,
                      UR_PLATFORM_INFO_ADAPTER),
    ur_platform_info_t);

TEST_P(urPlatformGetInfoTest, Success) {
    size_t size = 0;
    ur_platform_info_t info_type = getParam();
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urPlatformGetInfo(platform, info_type, 0, nullptr, &size), info_type);
    if (info_type == UR_PLATFORM_INFO_BACKEND) {
        ASSERT_EQ(size, sizeof(ur_platform_backend_t));
    } else {
        ASSERT_NE(size, 0);
    }
    std::vector<char> name(size);
    ASSERT_SUCCESS(
        urPlatformGetInfo(platform, info_type, size, name.data(), nullptr));
    switch (info_type) {
    case UR_PLATFORM_INFO_NAME:
    case UR_PLATFORM_INFO_VENDOR_NAME:
    case UR_PLATFORM_INFO_VERSION:
    case UR_PLATFORM_INFO_EXTENSIONS:
    case UR_PLATFORM_INFO_PROFILE: {
        ASSERT_EQ(size, std::strlen(name.data()) + 1);
        break;
    }
    case UR_PLATFORM_INFO_BACKEND: {
        ASSERT_EQ(size, sizeof(ur_platform_backend_t));
        break;
    }
    case UR_PLATFORM_INFO_ADAPTER: {
        auto queried_adapter =
            *reinterpret_cast<ur_adapter_handle_t *>(name.data());
        auto adapter_found =
            std::find(uur::PlatformEnvironment::instance->adapters.begin(),
                      uur::PlatformEnvironment::instance->adapters.end(),
                      queried_adapter);
        ASSERT_NE(adapter_found,
                  uur::AdapterEnvironment::instance->adapters.end());
        break;
    }
    default:
        break;
    }
}

TEST_P(urPlatformGetInfoTest, InvalidNullHandlePlatform) {
    size_t size = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urPlatformGetInfo(nullptr, getParam(), 0, nullptr, &size));
}

using urPlatformGetInfoNegativeTest = uur::urPlatformTest;
UUR_INSTANTIATE_PLATFORM_TEST_SUITE_P(urPlatformGetInfoNegativeTest);

TEST_P(urPlatformGetInfoNegativeTest, InvalidEnumerationPlatformInfoType) {
    size_t size = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urPlatformGetInfo(platform, UR_PLATFORM_INFO_FORCE_UINT32,
                                       0, nullptr, &size));
}

TEST_P(urPlatformGetInfoNegativeTest, InvalidSizeZero) {
    ur_platform_backend_t backend;
    ASSERT_EQ_RESULT(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND, 0,
                                       &backend, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urPlatformGetInfoNegativeTest, InvalidSizeSmall) {
    ur_platform_backend_t backend;
    ASSERT_EQ_RESULT(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                       sizeof(backend) - 1, &backend, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urPlatformGetInfoNegativeTest, InvalidNullPointerPropValue) {
    ur_platform_backend_t backend;
    ASSERT_EQ_RESULT(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                       sizeof(backend), nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urPlatformGetInfoNegativeTest, InvalidNullPointerPropSizeRet) {
    ASSERT_EQ_RESULT(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND, 0,
                                       nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
