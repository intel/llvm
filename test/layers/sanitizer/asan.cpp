/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan.cpp
 *
 */

#include <gtest/gtest.h>
#include <ur_api.h>

TEST(DeviceAsan, Initialization) {
    ur_result_t status;

    ur_loader_config_handle_t loaderConfig;
    status = urLoaderConfigCreate(&loaderConfig);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);
    status = urLoaderConfigEnableLayer(loaderConfig, "UR_LAYER_ASAN");
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    status = urLoaderInit(0, loaderConfig);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    ur_adapter_handle_t adapter;
    status = urAdapterGet(1, &adapter, nullptr);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    ur_platform_handle_t platform;
    status = urPlatformGet(&adapter, 1, 1, &platform, nullptr);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    ur_device_handle_t device;
    status = urDeviceGet(platform, UR_DEVICE_TYPE_DEFAULT, 1, &device, nullptr);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    ur_context_handle_t context;
    status = urContextCreate(1, &device, nullptr, &context);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    status = urContextRelease(context);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    status = urDeviceRelease(device);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    status = urAdapterRelease(adapter);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    status = urLoaderTearDown();
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    status = urLoaderConfigRelease(loaderConfig);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);
}

TEST(DeviceAsan, UnsupportedFeature) {
    ur_result_t status;

    ur_loader_config_handle_t loaderConfig;
    status = urLoaderConfigCreate(&loaderConfig);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);
    status = urLoaderConfigEnableLayer(loaderConfig, "UR_LAYER_ASAN");
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    status = urLoaderInit(0, loaderConfig);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    ur_adapter_handle_t adapter;
    status = urAdapterGet(1, &adapter, nullptr);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    ur_platform_handle_t platform;
    status = urPlatformGet(&adapter, 1, 1, &platform, nullptr);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    ur_device_handle_t device;
    status = urDeviceGet(platform, UR_DEVICE_TYPE_DEFAULT, 1, &device, nullptr);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    ur_context_handle_t context;
    status = urContextCreate(1, &device, nullptr, &context);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    // Check for explict unsupported features
    ur_bool_t isSupported;
    status = urDeviceGetInfo(device, UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT,
                             sizeof(isSupported), &isSupported, nullptr);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);
    ASSERT_EQ(isSupported, 0);

    status = urDeviceGetInfo(device, UR_DEVICE_INFO_COMMAND_BUFFER_SUPPORT_EXP,
                             sizeof(isSupported), &isSupported, nullptr);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);
    ASSERT_EQ(isSupported, 0);

    ur_device_command_buffer_update_capability_flags_t update_flag;
    status = urDeviceGetInfo(
        device, UR_DEVICE_INFO_COMMAND_BUFFER_UPDATE_CAPABILITIES_EXP,
        sizeof(update_flag), &update_flag, nullptr);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);
    ASSERT_EQ(update_flag, 0);

    status = urContextRelease(context);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    status = urDeviceRelease(device);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    status = urAdapterRelease(adapter);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    status = urLoaderTearDown();
    ASSERT_EQ(status, UR_RESULT_SUCCESS);

    status = urLoaderConfigRelease(loaderConfig);
    ASSERT_EQ(status, UR_RESULT_SUCCESS);
}
