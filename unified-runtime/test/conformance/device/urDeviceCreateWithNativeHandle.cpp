// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>
#include <uur/known_failure.h>

using urDeviceCreateWithNativeHandleTest = uur::urDeviceTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urDeviceCreateWithNativeHandleTest);

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

  // Based on the spec we can expect these to be equal.
  ASSERT_EQ(dev, device);

  uint32_t dev_id = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(dev, UR_DEVICE_INFO_TYPE, sizeof(uint32_t),
                                 &dev_id, nullptr));
}

TEST_P(urDeviceCreateWithNativeHandleTest,
       SuccessWithExplicitUnOwnedNativeHandle) {
  ur_native_handle_t native_handle = 0;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetNativeHandle(device, &native_handle));

  ur_device_handle_t dev = nullptr;
  ur_device_native_properties_t props{
      UR_STRUCTURE_TYPE_DEVICE_NATIVE_PROPERTIES, nullptr, false};
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceCreateWithNativeHandle(native_handle, adapter, &props, &dev));
  ASSERT_NE(dev, nullptr);
  ASSERT_EQ(dev, device);
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

TEST_P(urDeviceCreateWithNativeHandleTest, SubDeviceHandleEquality) {
  if (!uur::hasDevicePartitionSupport(device, UR_DEVICE_PARTITION_EQUALLY)) {
    GTEST_SKIP() << "Device: \'" << device
                 << "\' does not support partitioning equally.";
  }

  ur_device_partition_property_t property = uur::makePartitionEquallyDesc(2);
  ur_device_partition_properties_t properties{
      UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES,
      nullptr,
      &property,
      1,
  };

  std::vector<ur_device_handle_t> sub_devices(2);
  ASSERT_SUCCESS(
      urDevicePartition(device, &properties, 2, sub_devices.data(), nullptr));

  ur_native_handle_t native;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetNativeHandle(sub_devices[0], &native));

  ur_device_handle_t handle_a;
  ASSERT_SUCCESS(
      urDeviceCreateWithNativeHandle(native, adapter, nullptr, &handle_a));
  ur_device_handle_t handle_b;
  ASSERT_SUCCESS(
      urDeviceCreateWithNativeHandle(native, adapter, nullptr, &handle_b));

  ASSERT_EQ(handle_a, handle_b);
  ASSERT_NE(handle_a, nullptr);
}
