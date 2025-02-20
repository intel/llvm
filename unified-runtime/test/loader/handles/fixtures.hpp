// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_LOADER_CONFIG_TEST_FIXTURES_H
#define UR_LOADER_CONFIG_TEST_FIXTURES_H

#include "ur_api.h"
#include <gtest/gtest.h>
#include <ur_mock_helpers.hpp>

#ifndef ASSERT_SUCCESS
#define ASSERT_SUCCESS(ACTUAL) ASSERT_EQ(UR_RESULT_SUCCESS, ACTUAL)
#endif

ur_result_t replace_urPlatformGet(void *pParams) {
  const auto &params = *static_cast<ur_platform_get_params_t *>(pParams);

  if (*params.ppNumPlatforms) {
    **params.ppNumPlatforms = 1;
  }

  if (*params.pphPlatforms && *params.pNumEntries == 1) {
    **params.pphPlatforms = reinterpret_cast<ur_platform_handle_t>(0x1);
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t replace_urDeviceGetInfo(void *pParams) {
  const auto &params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_PLATFORM) {
    if (*params.ppPropSizeRet) {
      **params.ppPropSizeRet = sizeof(ur_platform_handle_t);
    }
    if (*params.ppPropValue) {
      **(reinterpret_cast<ur_platform_handle_t **>(params.ppPropValue)) =
          reinterpret_cast<ur_platform_handle_t>(0x1);
    }
  }
  return UR_RESULT_SUCCESS;
}

struct LoaderHandleTest : ::testing::Test {
  void SetUp() override {
    urLoaderInit(0, nullptr);
    mock::getCallbacks().set_replace_callback("urDeviceGetInfo",
                                              &replace_urDeviceGetInfo);
    mock::getCallbacks().set_replace_callback("urPlatformGet",
                                              &replace_urPlatformGet);
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
    mock::getCallbacks().resetCallbacks();
    urDeviceRelease(device);
    urAdapterRelease(adapter);
    urLoaderTearDown();
  }

  ur_adapter_handle_t adapter;
  ur_platform_handle_t platform;
  ur_device_handle_t device;
};

#endif
