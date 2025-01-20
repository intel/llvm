// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urHipGetDeviceNativeHandle = uur::urDeviceTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urHipGetDeviceNativeHandle);

TEST_P(urHipGetDeviceNativeHandle, Success) {
  ur_native_handle_t native_handle;
  ASSERT_SUCCESS(urDeviceGetNativeHandle(device, &native_handle));

  hipDevice_t hip_device;
  memcpy(&hip_device, &native_handle, sizeof(hip_device));

  char hip_device_name[256];
  ASSERT_SUCCESS_HIP(
      hipDeviceGetName(hip_device_name, sizeof(hip_device_name), hip_device));
}
