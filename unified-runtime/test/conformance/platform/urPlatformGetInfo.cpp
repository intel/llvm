// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include "uur/environment.h"
#include "uur/fixtures.h"
#include <cstring>

using urPlatformGetInfoTest = uur::urPlatformTest;
UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urPlatformGetInfoTest);

TEST_P(urPlatformGetInfoTest, SuccessName) {
  const ur_platform_info_t property_name = UR_PLATFORM_INFO_NAME;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urPlatformGetInfo(platform, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_GT(property_size, 0);

  std::vector<char> property_value(property_size, '\0');
  ASSERT_SUCCESS(urPlatformGetInfo(platform, property_name, property_size,
                                   property_value.data(), nullptr));

  ASSERT_TRUE(uur::stringPropertyIsValid(property_value.data(), property_size));
}

TEST_P(urPlatformGetInfoTest, SuccessVendorName) {
  const ur_platform_info_t property_name = UR_PLATFORM_INFO_VENDOR_NAME;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urPlatformGetInfo(platform, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_GT(property_size, 0);

  std::vector<char> property_value(property_size, '\0');
  ASSERT_SUCCESS(urPlatformGetInfo(platform, property_name, property_size,
                                   property_value.data(), nullptr));

  ASSERT_TRUE(uur::stringPropertyIsValid(property_value.data(), property_size));
}

TEST_P(urPlatformGetInfoTest, SuccessVersion) {
  const ur_platform_info_t property_name = UR_PLATFORM_INFO_VERSION;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urPlatformGetInfo(platform, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_GT(property_size, 0);

  std::vector<char> property_value(property_size, '\0');
  ASSERT_SUCCESS(urPlatformGetInfo(platform, property_name, property_size,
                                   property_value.data(), nullptr));

  ASSERT_TRUE(uur::stringPropertyIsValid(property_value.data(), property_size));
}

TEST_P(urPlatformGetInfoTest, SuccessExtensions) {
  const ur_platform_info_t property_name = UR_PLATFORM_INFO_EXTENSIONS;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urPlatformGetInfo(platform, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_GT(property_size, 0);

  std::vector<char> property_value(property_size, '\0');
  ASSERT_SUCCESS(urPlatformGetInfo(platform, property_name, property_size,
                                   property_value.data(), nullptr));

  ASSERT_TRUE(uur::stringPropertyIsValid(property_value.data(), property_size));
}

TEST_P(urPlatformGetInfoTest, SuccessProfile) {
  const ur_platform_info_t property_name = UR_PLATFORM_INFO_PROFILE;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urPlatformGetInfo(platform, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_GT(property_size, 0);

  std::vector<char> property_value(property_size, '\0');
  ASSERT_SUCCESS(urPlatformGetInfo(platform, property_name, property_size,
                                   property_value.data(), nullptr));

  ASSERT_TRUE(uur::stringPropertyIsValid(property_value.data(), property_size));
}

TEST_P(urPlatformGetInfoTest, SuccessBackend) {
  const ur_platform_info_t property_name = UR_PLATFORM_INFO_BACKEND;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urPlatformGetInfo(platform, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_backend_t));

  ur_backend_t property_value = UR_BACKEND_UNKNOWN;
  ASSERT_SUCCESS(urPlatformGetInfo(platform, property_name, property_size,
                                   &property_value, nullptr));

  ASSERT_TRUE(property_value >= UR_BACKEND_LEVEL_ZERO &&
              property_value <= UR_BACKEND_NATIVE_CPU);
}

TEST_P(urPlatformGetInfoTest, SuccessAdapter) {
  const ur_platform_info_t property_name = UR_PLATFORM_INFO_ADAPTER;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urPlatformGetInfo(platform, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_adapter_handle_t));

  ur_adapter_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urPlatformGetInfo(platform, property_name, property_size,
                                   &property_value, nullptr));

  auto adapter_found = std::find(
      uur::PlatformEnvironment::instance->adapters.begin(),
      uur::PlatformEnvironment::instance->adapters.end(), property_value);
  ASSERT_NE(adapter_found, uur::AdapterEnvironment::instance->adapters.end());
}

TEST_P(urPlatformGetInfoTest, SuccessRoundtripAdapter) {
  const ur_platform_info_t property_name = UR_PLATFORM_INFO_ADAPTER;
  size_t property_size = sizeof(ur_adapter_handle_t);

  ur_adapter_handle_t adapter = nullptr;
  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urPlatformGetInfo(platform, UR_PLATFORM_INFO_ADAPTER,
                        sizeof(ur_adapter_handle_t), &adapter, nullptr),
      UR_PLATFORM_INFO_ADAPTER);

  ur_native_handle_t native_platform;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urPlatformGetNativeHandle(platform, &native_platform));

  ur_platform_handle_t from_native_platform;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urPlatformCreateWithNativeHandle(
      native_platform, adapter, nullptr, &from_native_platform));

  ur_adapter_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urPlatformGetInfo(from_native_platform, property_name,
                                   property_size, &property_value, nullptr));

  ASSERT_EQ(adapter, property_value);
}

TEST_P(urPlatformGetInfoTest, InvalidNullHandlePlatform) {
  size_t property_size = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urPlatformGetInfo(nullptr, UR_PLATFORM_INFO_NAME, 0, nullptr,
                                     &property_size));
}

TEST_P(urPlatformGetInfoTest, InvalidEnumerationPlatformInfoType) {
  size_t property_size = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urPlatformGetInfo(platform, UR_PLATFORM_INFO_FORCE_UINT32, 0,
                                     nullptr, &property_size));
}

TEST_P(urPlatformGetInfoTest, InvalidSizeZero) {
  ur_backend_t property_value = UR_BACKEND_UNKNOWN;
  ASSERT_EQ_RESULT(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND, 0,
                                     &property_value, nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urPlatformGetInfoTest, InvalidSizeSmall) {
  ur_backend_t property_value = UR_BACKEND_UNKNOWN;
  ASSERT_EQ_RESULT(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                     sizeof(property_value) - 1,
                                     &property_value, nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urPlatformGetInfoTest, InvalidNullPointerPropValue) {
  ur_backend_t property_value = UR_BACKEND_UNKNOWN;
  ASSERT_EQ_RESULT(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                     sizeof(property_value), nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urPlatformGetInfoTest, InvalidNullPointerPropSizeRet) {
  ASSERT_EQ_RESULT(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND, 0,
                                     nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
