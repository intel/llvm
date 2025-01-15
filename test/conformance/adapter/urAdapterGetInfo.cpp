// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

#include <cstring>

using urAdapterGetInfoTest = uur::urAdapterTest;

UUR_INSTANTIATE_ADAPTER_TEST_SUITE_P(urAdapterGetInfoTest);

TEST_P(urAdapterGetInfoTest, SuccessBackend) {
  ur_adapter_info_t property_name = UR_ADAPTER_INFO_BACKEND;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urAdapterGetInfo(adapter, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_adapter_backend_t));

  ur_adapter_backend_t backend = UR_ADAPTER_BACKEND_UNKNOWN;
  ASSERT_SUCCESS(urAdapterGetInfo(adapter, property_name, property_size,
                                  &backend, nullptr));

  ASSERT_TRUE(backend >= UR_ADAPTER_BACKEND_LEVEL_ZERO &&
              backend <= UR_ADAPTER_BACKEND_NATIVE_CPU);
}

TEST_P(urAdapterGetInfoTest, SuccessReferenceCount) {
  ur_adapter_info_t property_name = UR_ADAPTER_INFO_REFERENCE_COUNT;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urAdapterGetInfo(adapter, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t reference_count = 0;
  ASSERT_SUCCESS(urAdapterGetInfo(adapter, property_name, property_size,
                                  &reference_count, nullptr));
  ASSERT_GE(reference_count, 0);
}

TEST_P(urAdapterGetInfoTest, SuccessVersion) {
  ur_adapter_info_t property_name = UR_ADAPTER_INFO_VERSION;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urAdapterGetInfo(adapter, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t returned_version = 46;
  ASSERT_SUCCESS(urAdapterGetInfo(adapter, property_name, property_size,
                                  &returned_version, nullptr));
  ASSERT_NE(42, returned_version);
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
  ur_adapter_backend_t backend = UR_ADAPTER_BACKEND_UNKNOWN;
  ASSERT_EQ_RESULT(
      urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND, 0, &backend, nullptr),
      UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urAdapterGetInfoTest, InvalidSizeSmall) {
  ur_adapter_backend_t backend = UR_ADAPTER_BACKEND_UNKNOWN;
  ASSERT_EQ_RESULT(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND,
                                    sizeof(backend) - 1, &backend, nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urAdapterGetInfoTest, InvalidNullPointerPropValue) {
  ur_adapter_backend_t backend = UR_ADAPTER_BACKEND_UNKNOWN;
  ASSERT_EQ_RESULT(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND,
                                    sizeof(backend), nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urAdapterGetInfoTest, InvalidNullPointerPropSizeRet) {
  ASSERT_EQ_RESULT(
      urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND, 0, nullptr, nullptr),
      UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
