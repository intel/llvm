// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

#include <cstring>

using urAdapterGetInfoTest = uur::urAdapterTest;

UUR_INSTANTIATE_ADAPTER_TEST_SUITE(urAdapterGetInfoTest);

TEST_P(urAdapterGetInfoTest, SuccessBackend) {
  const ur_adapter_info_t property_name = UR_ADAPTER_INFO_BACKEND;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urAdapterGetInfo(adapter, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_backend_t));

  ur_backend_t property_value = UR_BACKEND_UNKNOWN;
  ASSERT_SUCCESS(urAdapterGetInfo(adapter, property_name, property_size,
                                  &property_value, nullptr));

  ASSERT_TRUE(property_value >= UR_BACKEND_LEVEL_ZERO &&
              property_value <= UR_BACKEND_OFFLOAD);
}

TEST_P(urAdapterGetInfoTest, SuccessReferenceCount) {
  const ur_adapter_info_t property_name = UR_ADAPTER_INFO_REFERENCE_COUNT;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urAdapterGetInfo(adapter, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urAdapterGetInfo(adapter, property_name,
                                              property_size, &property_value,
                                              nullptr),
                             property_value);
  ASSERT_GE(property_value, 0);
}

TEST_P(urAdapterGetInfoTest, SuccessVersion) {
  const ur_adapter_info_t property_name = UR_ADAPTER_INFO_VERSION;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urAdapterGetInfo(adapter, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urAdapterGetInfo(adapter, property_name,
                                              property_size, &property_value,
                                              nullptr),
                             property_value);
}

TEST_P(urAdapterGetInfoTest, InvalidNullHandleAdapter) {
  size_t property_size = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urAdapterGetInfo(nullptr, UR_ADAPTER_INFO_BACKEND, 0,
                                    nullptr, &property_size));
}

TEST_P(urAdapterGetInfoTest, InvalidEnumerationAdapterInfoType) {
  size_t property_size = 0;

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urAdapterGetInfo(adapter, UR_ADAPTER_INFO_FORCE_UINT32, 0,
                                    nullptr, &property_size));
}

TEST_P(urAdapterGetInfoTest, InvalidSizeZero) {
  ur_backend_t backend = UR_BACKEND_UNKNOWN;
  ASSERT_EQ_RESULT(
      urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND, 0, &backend, nullptr),
      UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urAdapterGetInfoTest, InvalidSizeSmall) {
  ur_backend_t backend = UR_BACKEND_UNKNOWN;
  ASSERT_EQ_RESULT(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND,
                                    sizeof(backend) - 1, &backend, nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urAdapterGetInfoTest, InvalidNullPointerPropValue) {
  const ur_backend_t backend = UR_BACKEND_UNKNOWN;
  ASSERT_EQ_RESULT(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND,
                                    sizeof(backend), nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urAdapterGetInfoTest, InvalidNullPointerPropSizeRet) {
  ASSERT_EQ_RESULT(
      urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND, 0, nullptr, nullptr),
      UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
