// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urCudaDeviceCreateWithNativeHandle = uur::urPlatformTest;

TEST_F(urCudaDeviceCreateWithNativeHandle, Success) {
    // get a device from cuda
    int nCudaDevices;
    ASSERT_SUCCESS_CUDA(cuDeviceGetCount(&nCudaDevices));
    ASSERT_GT(nCudaDevices, 0);
    CUdevice cudaDevice;
    ASSERT_SUCCESS_CUDA(cuDeviceGet(&cudaDevice, 0));

    ur_native_handle_t nativeCuda = reinterpret_cast<ur_native_handle_t>(
        static_cast<std::uintptr_t>(cudaDevice));
    ur_device_handle_t urDevice;
    ASSERT_SUCCESS(urDeviceCreateWithNativeHandle(nativeCuda, platform, nullptr,
                                                  &urDevice));
}
