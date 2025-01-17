/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
 * LLVM-exception
 *
 * @file codeloc.cpp
 *
 */

#include "uur/raii.h"
#include <gtest/gtest.h>
#include <ur_api.h>

#include <ur_mock_helpers.hpp>

TEST(Mock, NullHandle) {
  ASSERT_EQ(urLoaderConfigSetMockingEnabled(nullptr, true),
            UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST(Mock, DefaultBehavior) {
  uur::raii::LoaderConfig loader_config;
  ASSERT_EQ(urLoaderConfigCreate(loader_config.ptr()), UR_RESULT_SUCCESS);
  ASSERT_EQ(urLoaderConfigSetMockingEnabled(loader_config, true),
            UR_RESULT_SUCCESS);
  ASSERT_EQ(urLoaderInit(0, loader_config), UR_RESULT_SUCCESS);

  // Set up as far as device and check we're getting sensible, different
  // handles created.
  ur_adapter_handle_t adapter = nullptr;
  ur_platform_handle_t platform = nullptr;
  ur_device_handle_t device = nullptr;

  ASSERT_EQ(urAdapterGet(1, &adapter, nullptr), UR_RESULT_SUCCESS);
  ASSERT_EQ(urPlatformGet(&adapter, 1, 1, &platform, nullptr),
            UR_RESULT_SUCCESS);
  ASSERT_EQ(urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 1, &device, nullptr),
            UR_RESULT_SUCCESS);

  ASSERT_NE(adapter, nullptr);
  ASSERT_NE(platform, nullptr);
  ASSERT_NE(device, nullptr);

  ASSERT_NE(static_cast<void *>(adapter), static_cast<void *>(platform));
  ASSERT_NE(static_cast<void *>(adapter), static_cast<void *>(device));
  ASSERT_NE(static_cast<void *>(platform), static_cast<void *>(device));

  ASSERT_EQ(urDeviceRelease(device), UR_RESULT_SUCCESS);
}

ur_result_t beforeUrAdapterGet(void *pParams) {
  auto params = reinterpret_cast<ur_adapter_get_params_t *>(pParams);
  ur_adapter_handle_t preInitAdapter =
      reinterpret_cast<ur_adapter_handle_t>(uintptr_t(0xF00DCAFE));
  EXPECT_EQ(**params->pphAdapters, preInitAdapter);
  return UR_RESULT_SUCCESS;
}

ur_result_t replaceUrAdapterGet(void *pParams) {
  auto params = reinterpret_cast<ur_adapter_get_params_t *>(pParams);
  **params->pphAdapters =
      reinterpret_cast<ur_adapter_handle_t>(uintptr_t(0xDEADBEEF));
  return UR_RESULT_SUCCESS;
}

void checkPostInitAdapter(ur_adapter_handle_t adapter) {
  ur_adapter_handle_t postInitAdapter =
      reinterpret_cast<ur_adapter_handle_t>(uintptr_t(0xDEADBEEF));
  ASSERT_EQ(adapter, postInitAdapter);
}

ur_result_t afterUrAdapterGet(void *pParams) {
  auto params = reinterpret_cast<ur_adapter_get_params_t *>(pParams);
  checkPostInitAdapter(**params->pphAdapters);
  return UR_RESULT_SUCCESS;
}

TEST(Mock, Callbacks) {
  uur::raii::LoaderConfig loader_config;
  ASSERT_EQ(urLoaderConfigCreate(loader_config.ptr()), UR_RESULT_SUCCESS);
  ASSERT_EQ(urLoaderConfigSetMockingEnabled(loader_config, true),
            UR_RESULT_SUCCESS);
  ASSERT_EQ(urLoaderInit(0, loader_config), UR_RESULT_SUCCESS);

  // This callback is set up to check *phAdapters is still the pre-call
  // init value we set below
  mock::getCallbacks().set_before_callback("urAdapterGet", &beforeUrAdapterGet);

  // This callback is set up to return a distinct test value in phAdapters
  // rather than the default generic handle
  mock::getCallbacks().set_replace_callback("urAdapterGet",
                                            &replaceUrAdapterGet);

  // This callback is set up to check our replace callback did its job
  mock::getCallbacks().set_after_callback("urAdapterGet", &afterUrAdapterGet);

  ur_adapter_handle_t adapter =
      reinterpret_cast<ur_adapter_handle_t>(uintptr_t(0xF00DCAFE));
  ASSERT_EQ(urAdapterGet(1, &adapter, nullptr), UR_RESULT_SUCCESS);
}
