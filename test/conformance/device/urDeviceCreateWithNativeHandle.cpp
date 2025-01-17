// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception
#include <uur/fixtures.h>
#include <uur/known_failure.h>

using urDeviceCreateWithNativeHandleTest = uur::urDeviceTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urDeviceCreateWithNativeHandleTest);

TEST_P(urDeviceCreateWithNativeHandleTest, Success) {
  ur_native_handle_t native_handle = 0;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetNativeHandle(device, &native_handle));

  // We cannot assume anything about a native_handle, not even if it's
  // `nullptr` since this could be a valid representation within a backend.
  // We can however convert the native_handle back into a unified-runtime handle
  // and perform some query on it to verify that it works.
  ur_device_handle_t dev = nullptr;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceCreateWithNativeHandle(native_handle, adapter, nullptr, &dev));
  ASSERT_NE(dev, nullptr);

  uint32_t dev_id = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(dev, UR_DEVICE_INFO_TYPE, sizeof(uint32_t),
                                 &dev_id, nullptr));
}

TEST_P(urDeviceCreateWithNativeHandleTest, SuccessWithOwnedNativeHandle) {
  ur_native_handle_t native_handle = 0;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetNativeHandle(device, &native_handle));

  ur_device_handle_t dev = nullptr;
  ur_device_native_properties_t props{
      UR_STRUCTURE_TYPE_DEVICE_NATIVE_PROPERTIES, nullptr, true};
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceCreateWithNativeHandle(native_handle, adapter, &props, &dev));
  ASSERT_NE(dev, nullptr);
}

TEST_P(urDeviceCreateWithNativeHandleTest, SuccessWithUnOwnedNativeHandle) {
  ur_native_handle_t native_handle = 0;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetNativeHandle(device, &native_handle));

  ur_device_handle_t dev = nullptr;
  ur_device_native_properties_t props{
      UR_STRUCTURE_TYPE_DEVICE_NATIVE_PROPERTIES, nullptr, false};
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceCreateWithNativeHandle(native_handle, adapter, &props, &dev));
  ASSERT_NE(dev, nullptr);
}

TEST_P(urDeviceCreateWithNativeHandleTest, InvalidNullHandlePlatform) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  ur_native_handle_t native_handle = 0;
  ASSERT_SUCCESS(urDeviceGetNativeHandle(device, &native_handle));

  ur_device_handle_t dev = nullptr;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_HANDLE,
      urDeviceCreateWithNativeHandle(native_handle, nullptr, nullptr, &dev));
}

TEST_P(urDeviceCreateWithNativeHandleTest, InvalidNullPointerDevice) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  ur_native_handle_t native_handle = 0;
  ASSERT_SUCCESS(urDeviceGetNativeHandle(device, &native_handle));

  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_POINTER,
      urDeviceCreateWithNativeHandle(native_handle, adapter, nullptr, nullptr));
}
