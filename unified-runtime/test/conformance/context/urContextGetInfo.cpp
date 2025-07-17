// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urContextGetInfoTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urContextGetInfoTest);

TEST_P(urContextGetInfoTest, SuccessNumDevices) {
  const ur_context_info_t property_name = UR_CONTEXT_INFO_NUM_DEVICES;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urContextGetInfo(context, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urContextGetInfo(context, property_name,
                                              property_size, &property_value,
                                              nullptr),
                             property_value);
}

TEST_P(urContextGetInfoTest, SuccessDevices) {
  const ur_context_info_t property_name = UR_CONTEXT_INFO_DEVICES;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urContextGetInfo(context, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_NE(property_size, 0);

  ur_device_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urContextGetInfo(context, property_name, property_size,
                                  &property_value, nullptr));

  size_t devices_count = property_size / sizeof(ur_device_handle_t);
  ASSERT_EQ(devices_count, 1);
  ASSERT_EQ(property_value, device);
}

TEST_P(urContextGetInfoTest, SuccessRoundtripDevices) {
  const ur_context_info_t property_name = UR_CONTEXT_INFO_DEVICES;
  size_t property_size = sizeof(ur_device_handle_t);

  ur_native_handle_t native_context;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urContextGetNativeHandle(context, &native_context));

  ur_context_handle_t from_native_context;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urContextCreateWithNativeHandle(
      native_context, adapter, 1, &device, nullptr, &from_native_context));

  ur_device_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urContextGetInfo(from_native_context, property_name,
                                  property_size, &property_value, nullptr));

  size_t devices_count = property_size / sizeof(ur_device_handle_t);
  ASSERT_EQ(devices_count, 1);
  ASSERT_EQ(property_value, device);
}

TEST_P(urContextGetInfoTest, SuccessUSMMemCpy2DSupport) {
  const ur_context_info_t property_name = UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urContextGetInfo(context, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));
}

TEST_P(urContextGetInfoTest, SuccessUSMFill2DSupport) {
  const ur_context_info_t property_name = UR_CONTEXT_INFO_USM_FILL2D_SUPPORT;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urContextGetInfo(context, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));
}

TEST_P(urContextGetInfoTest, SuccessReferenceCount) {
  const ur_context_info_t property_name = UR_CONTEXT_INFO_REFERENCE_COUNT;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urContextGetInfo(context, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urContextGetInfo(context, property_name,
                                              property_size, &property_value,
                                              nullptr),
                             property_value);
  ASSERT_GT(property_value, 0U);
}

TEST_P(urContextGetInfoTest, InvalidNullHandleContext) {
  uint32_t nDevices = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urContextGetInfo(nullptr, UR_CONTEXT_INFO_NUM_DEVICES,
                                    sizeof(uint32_t), &nDevices, nullptr));
}

TEST_P(urContextGetInfoTest, InvalidEnumeration) {
  uint32_t nDevices = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urContextGetInfo(context, UR_CONTEXT_INFO_FORCE_UINT32,
                                    sizeof(uint32_t), &nDevices, nullptr));
}

TEST_P(urContextGetInfoTest, InvalidSizePropSize) {
  uint32_t nDevices = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urContextGetInfo(context, UR_CONTEXT_INFO_NUM_DEVICES, 0,
                                    &nDevices, nullptr));
}

TEST_P(urContextGetInfoTest, InvalidSizePropSizeSmall) {
  uint32_t nDevices = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urContextGetInfo(context, UR_CONTEXT_INFO_NUM_DEVICES,
                                    sizeof(nDevices) - 1, &nDevices, nullptr));
}

TEST_P(urContextGetInfoTest, InvalidNullPointerPropValue) {
  uint32_t nDevices = 0;
  ASSERT_EQ_RESULT(urContextGetInfo(context, UR_CONTEXT_INFO_NUM_DEVICES,
                                    sizeof(nDevices), nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urContextGetInfoTest, InvalidNullPointerPropSizeRet) {
  ASSERT_EQ_RESULT(urContextGetInfo(context, UR_CONTEXT_INFO_NUM_DEVICES, 0,
                                    nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
