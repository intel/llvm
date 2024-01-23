// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_LOADER_CONFIG_TEST_FIXTURES_H
#define UR_LOADER_CONFIG_TEST_FIXTURES_H

#include "ur_api.h"
#include <gtest/gtest.h>

#ifndef ASSERT_SUCCESS
#define ASSERT_SUCCESS(ACTUAL) ASSERT_EQ(UR_RESULT_SUCCESS, ACTUAL)
#endif

struct LoaderHandleTest : ::testing::Test {
    void SetUp() override {
        urLoaderInit(0, nullptr);
        uint32_t nadapters = 0;
        adapter = nullptr;
        ASSERT_SUCCESS(urAdapterGet(1, &adapter, &nadapters));
        ASSERT_NE(adapter, nullptr);
        uint32_t nplatforms = 0;
        platform = nullptr;
        ASSERT_SUCCESS(urPlatformGet(&adapter, 1, 1, &platform, &nplatforms));
        ASSERT_NE(platform, nullptr);
        uint32_t ndevices;
        device = nullptr;
        ASSERT_SUCCESS(
            urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 1, &device, &ndevices));
        ASSERT_NE(device, nullptr);
    }

    void TearDown() override {
        urDeviceRelease(device);
        urAdapterRelease(adapter);
        urLoaderTearDown();
    }

    ur_adapter_handle_t adapter;
    ur_platform_handle_t platform;
    ur_device_handle_t device;
};

#endif
