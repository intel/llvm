// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include "uur/environment.h"
#include "uur/fixtures.h"
#include <cstring>

using urPlatformGetInfoTest = uur::urPlatformTest;
UUR_INSTANTIATE_PLATFORM_TEST_SUITE_P(urPlatformGetInfoTest);

TEST_P(urPlatformGetInfoTest, SuccessName) {
    ur_platform_info_t property_name = UR_PLATFORM_INFO_NAME;
    size_t property_size = 0;

    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urPlatformGetInfo(platform, property_name, 0, nullptr, &property_size),
        property_name);
    ASSERT_GT(property_size, 0);

    std::vector<char> returned_name(property_size);
    ASSERT_SUCCESS(urPlatformGetInfo(platform, property_name, property_size,
                                     returned_name.data(), nullptr));

    ASSERT_EQ(property_size, returned_name.size());
}

TEST_P(urPlatformGetInfoTest, SuccessVendorName) {
    ur_platform_info_t property_name = UR_PLATFORM_INFO_VENDOR_NAME;
    size_t property_size = 0;

    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urPlatformGetInfo(platform, property_name, 0, nullptr, &property_size),
        property_name);
    ASSERT_GT(property_size, 0);

    std::vector<char> returned_vendor_name(property_size);
    ASSERT_SUCCESS(urPlatformGetInfo(platform, property_name, property_size,
                                     returned_vendor_name.data(), nullptr));

    ASSERT_EQ(property_size, returned_vendor_name.size());
}

TEST_P(urPlatformGetInfoTest, SuccessVersion) {
    ur_platform_info_t property_name = UR_PLATFORM_INFO_VERSION;
    size_t property_size = 0;

    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urPlatformGetInfo(platform, property_name, 0, nullptr, &property_size),
        property_name);
    ASSERT_GT(property_size, 0);

    std::vector<char> returned_version(property_size);
    ASSERT_SUCCESS(urPlatformGetInfo(platform, property_name, property_size,
                                     returned_version.data(), nullptr));

    ASSERT_EQ(property_size, returned_version.size());
}

TEST_P(urPlatformGetInfoTest, SuccessExtensions) {
    ur_platform_info_t property_name = UR_PLATFORM_INFO_EXTENSIONS;
    size_t property_size = 0;

    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urPlatformGetInfo(platform, property_name, 0, nullptr, &property_size),
        property_name);
    ASSERT_GT(property_size, 0);

    std::vector<char> returned_extensions(property_size);
    ASSERT_SUCCESS(urPlatformGetInfo(platform, property_name, property_size,
                                     returned_extensions.data(), nullptr));

    ASSERT_EQ(property_size, returned_extensions.size());
}

TEST_P(urPlatformGetInfoTest, SuccessProfile) {
    ur_platform_info_t property_name = UR_PLATFORM_INFO_PROFILE;
    size_t property_size = 0;

    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urPlatformGetInfo(platform, property_name, 0, nullptr, &property_size),
        property_name);
    ASSERT_GT(property_size, 0);

    std::vector<char> returned_profile(property_size);
    ASSERT_SUCCESS(urPlatformGetInfo(platform, property_name, property_size,
                                     returned_profile.data(), nullptr));

    ASSERT_EQ(property_size, returned_profile.size());
}

TEST_P(urPlatformGetInfoTest, SuccessBackend) {
    ur_platform_info_t property_name = UR_PLATFORM_INFO_BACKEND;
    size_t property_size = 0;

    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urPlatformGetInfo(platform, property_name, 0, nullptr, &property_size),
        property_name);
    ASSERT_EQ(property_size, sizeof(ur_platform_backend_t));

    ur_platform_backend_t returned_backend = UR_PLATFORM_BACKEND_UNKNOWN;
    ASSERT_SUCCESS(urPlatformGetInfo(platform, property_name, property_size,
                                     &returned_backend, nullptr));

    ASSERT_TRUE(returned_backend >= UR_PLATFORM_BACKEND_LEVEL_ZERO &&
                returned_backend <= UR_PLATFORM_BACKEND_NATIVE_CPU);
}

TEST_P(urPlatformGetInfoTest, SuccessAdapter) {
    ur_platform_info_t property_name = UR_PLATFORM_INFO_ADAPTER;
    size_t property_size = 0;

    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urPlatformGetInfo(platform, property_name, 0, nullptr, &property_size),
        property_name);
    ASSERT_EQ(property_size, sizeof(ur_adapter_handle_t));

    ur_adapter_handle_t returned_adapter = nullptr;
    ASSERT_SUCCESS(urPlatformGetInfo(platform, property_name, property_size,
                                     &returned_adapter, nullptr));

    auto adapter_found = std::find(
        uur::PlatformEnvironment::instance->adapters.begin(),
        uur::PlatformEnvironment::instance->adapters.end(), returned_adapter);
    ASSERT_NE(adapter_found, uur::AdapterEnvironment::instance->adapters.end());
}

TEST_P(urPlatformGetInfoTest, InvalidNullHandlePlatform) {
    size_t size = 0;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urPlatformGetInfo(nullptr, UR_PLATFORM_INFO_NAME, 0, nullptr, &size));
}

TEST_P(urPlatformGetInfoTest, InvalidEnumerationPlatformInfoType) {
    size_t size = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urPlatformGetInfo(platform, UR_PLATFORM_INFO_FORCE_UINT32,
                                       0, nullptr, &size));
}

TEST_P(urPlatformGetInfoTest, InvalidSizeZero) {
    ur_platform_backend_t backend = UR_PLATFORM_BACKEND_UNKNOWN;
    ASSERT_EQ_RESULT(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND, 0,
                                       &backend, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urPlatformGetInfoTest, InvalidSizeSmall) {
    ur_platform_backend_t backend = UR_PLATFORM_BACKEND_UNKNOWN;
    ASSERT_EQ_RESULT(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                       sizeof(backend) - 1, &backend, nullptr),
                     UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urPlatformGetInfoTest, InvalidNullPointerPropValue) {
    ur_platform_backend_t backend = UR_PLATFORM_BACKEND_UNKNOWN;
    ASSERT_EQ_RESULT(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                       sizeof(backend), nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urPlatformGetInfoTest, InvalidNullPointerPropSizeRet) {
    ASSERT_EQ_RESULT(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND, 0,
                                       nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
