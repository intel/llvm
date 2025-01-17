// Copyright (C) 2023-2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#ifndef UR_VALIDATION_TEST_HELPERS_H
#define UR_VALIDATION_TEST_HELPERS_H

#include <atomic>
#include <gtest/gtest.h>

#include <ur_api.h>
#include <ur_mock_helpers.hpp>

struct urTest : ::testing::Test {

  void SetUp() override {
    ASSERT_EQ(urLoaderConfigCreate(&loader_config), UR_RESULT_SUCCESS);
    ASSERT_EQ(
        urLoaderConfigEnableLayer(loader_config, "UR_LAYER_FULL_VALIDATION"),
        UR_RESULT_SUCCESS);
    ur_device_init_flags_t device_flags = 0;
    ASSERT_EQ(urLoaderInit(device_flags, loader_config), UR_RESULT_SUCCESS);
  }

  void TearDown() override {
    if (loader_config) {
      ASSERT_EQ(urLoaderConfigRelease(loader_config), UR_RESULT_SUCCESS);
    }
    ASSERT_EQ(urLoaderTearDown(), UR_RESULT_SUCCESS);
  }

  ur_loader_config_handle_t loader_config = nullptr;
};

struct valAdaptersTest : urTest {

  void SetUp() override {
    urTest::SetUp();

    uint32_t adapter_count;
    ASSERT_EQ(urAdapterGet(0, nullptr, &adapter_count), UR_RESULT_SUCCESS);
    ASSERT_GT(adapter_count, 0);
    adapters.resize(adapter_count);
    ASSERT_EQ(urAdapterGet(adapter_count, adapters.data(), nullptr),
              UR_RESULT_SUCCESS);
  }

  void TearDown() override {
    for (auto adapter : adapters) {
      ASSERT_EQ(urAdapterRelease(adapter), UR_RESULT_SUCCESS);
    }
    urTest::TearDown();
  }

  std::vector<ur_adapter_handle_t> adapters;
};

struct valAdapterTest : valAdaptersTest {

  void SetUp() override {
    valAdaptersTest::SetUp();
    adapter = adapters[0]; // TODO - which to choose?
  }

  ur_adapter_handle_t adapter;
};

struct valPlatformsTest : valAdaptersTest {

  void SetUp() override {
    valAdaptersTest::SetUp();

    uint32_t count;
    ASSERT_EQ(urPlatformGet(adapters.data(),
                            static_cast<uint32_t>(adapters.size()), 0, nullptr,
                            &count),
              UR_RESULT_SUCCESS);
    ASSERT_GT(count, 0);
    platforms.resize(count);
    ASSERT_EQ(urPlatformGet(adapters.data(),
                            static_cast<uint32_t>(adapters.size()), count,
                            platforms.data(), nullptr),
              UR_RESULT_SUCCESS);
  }

  std::vector<ur_platform_handle_t> platforms;
};

struct valPlatformTest : valPlatformsTest {

  void SetUp() override {
    valPlatformsTest::SetUp();
    ASSERT_GE(platforms.size(), 1);
    platform = platforms[0]; // TODO - which to choose?
  }

  ur_platform_handle_t platform;
};

struct valAllDevicesTest : valPlatformTest {

  void SetUp() override {
    valPlatformTest::SetUp();

    uint32_t count = 0;
    if (urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count) ||
        count == 0) {
      FAIL() << "Failed to get devices";
    }

    devices.resize(count);
    if (urDeviceGet(platform, UR_DEVICE_TYPE_ALL, count, devices.data(),
                    nullptr)) {
      FAIL() << "Failed to get devices";
    }
  }

  void TearDown() override {
    for (auto device : devices) {
      ASSERT_EQ(urDeviceRelease(device), UR_RESULT_SUCCESS);
    }
    valPlatformTest::TearDown();
  }

  std::vector<ur_device_handle_t> devices;
};

// We use this to avoid segfaults in the mock adapter when we're doing stuff
// like double releases in the leak detection tests.
inline ur_result_t genericSuccessCallback(void *) { return UR_RESULT_SUCCESS; }

// This returns valid (non-null) handles that we can safely leak.
inline ur_result_t fakeContext_urContextCreate(void *pParams) {
  static std::atomic_int handle = 42;
  const auto &params = *static_cast<ur_context_create_params_t *>(pParams);
  // There are two casts because windows doesn't implicitly extend the 32 bit
  // result of atomic_int::operator++.
  **params.pphContext =
      reinterpret_cast<ur_context_handle_t>(static_cast<uintptr_t>(handle++));
  return UR_RESULT_SUCCESS;
}

struct valDeviceTest : valAllDevicesTest {

  void SetUp() override {
    valAllDevicesTest::SetUp();
    ASSERT_GE(devices.size(), 1);
    device = devices[0];
    mock::getCallbacks().set_replace_callback("urContextRetain",
                                              &genericSuccessCallback);
    mock::getCallbacks().set_replace_callback("urContextRelease",
                                              &genericSuccessCallback);
    mock::getCallbacks().set_replace_callback("urContextCreate",
                                              &fakeContext_urContextCreate);
  }

  void TearDown() override {
    mock::getCallbacks().resetCallbacks();
    valAllDevicesTest::TearDown();
  }
  ur_device_handle_t device;
};

struct valDeviceTestMultithreaded : valDeviceTest,
                                    public ::testing::WithParamInterface<int> {
  void SetUp() override {
    valDeviceTest::SetUp();

    threadCount = GetParam();
  }

  int threadCount;
};

#endif // UR_VALIDATION_TEST_HELPERS_H
