// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include "fixtures.h"

using urCudaGetDeviceNativeHandle = uur::urDeviceTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urCudaGetDeviceNativeHandle);

TEST_P(urCudaGetDeviceNativeHandle, Success) {
  ur_native_handle_t native_handle;
  ASSERT_SUCCESS(urDeviceGetNativeHandle(device, &native_handle));

  CUdevice cuda_device;
  memcpy(&cuda_device, &native_handle, sizeof(cuda_device));

  char cuda_device_name[256];
  ASSERT_SUCCESS_CUDA(
      cuDeviceGetName(cuda_device_name, sizeof(cuda_device_name), cuda_device));
}
