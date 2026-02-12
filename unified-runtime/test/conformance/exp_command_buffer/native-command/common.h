// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once
#include "../fixtures.h"

namespace uur {
namespace command_buffer {

struct urCommandBufferNativeAppendTest : uur::urQueueTest {
  virtual void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::SetUp());

    UUR_RETURN_ON_FATAL_FAILURE(checkCommandBufferSupport(device));

    UUR_KNOWN_FAILURE_ON(uur::LevelZeroV2{});

    // Create a static command-buffer
    ur_exp_command_buffer_desc_t desc{UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC,
                                      nullptr, false, false, false};
    ASSERT_SUCCESS(
        urCommandBufferCreateExp(context, device, &desc, &command_buffer));

    ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                     sizeof(backend), &backend, nullptr));

    host_vec = std::vector<int>(global_size, 0);
    ASSERT_EQ(host_vec.size(), global_size);

    ur_device_usm_access_capability_flags_t device_usm_support;
    ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_USM_DEVICE_SUPPORT,
                                   sizeof(device_usm_support),
                                   &device_usm_support, nullptr));
    if (0 != device_usm_support) {
      ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                      allocation_size, &src_device_ptr));
      ASSERT_NE(src_device_ptr, nullptr);
      ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                      allocation_size, &dst_device_ptr));
      ASSERT_NE(dst_device_ptr, nullptr);
    }
  }

  virtual void TearDown() override {
    if (command_buffer) {
      EXPECT_SUCCESS(urCommandBufferReleaseExp(command_buffer));
    }

    if (src_device_ptr) {
      EXPECT_SUCCESS(urUSMFree(context, src_device_ptr));
    }

    if (dst_device_ptr) {
      EXPECT_SUCCESS(urUSMFree(context, dst_device_ptr));
    }
    UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::TearDown());
  }

  ur_backend_t backend{};
  ur_exp_command_buffer_handle_t command_buffer = nullptr;
  static constexpr int val = 42;
  static constexpr uint32_t global_size = 128;
  std::vector<int> host_vec;
  void *src_device_ptr = nullptr;
  void *dst_device_ptr = nullptr;
  static constexpr size_t allocation_size = sizeof(val) * global_size;
};

} // namespace command_buffer
} // namespace uur
