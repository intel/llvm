// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

struct InvalidCreationTest : uur::urContextTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urContextTest::SetUp());
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::command_buffer::checkCommandBufferSupport(device));
  }
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(InvalidCreationTest);

// Check correct error is reported when trying to create a
// command-buffer with update enabled on a device that doesn't
// support it.
TEST_P(InvalidCreationTest, Update) {
  ur_device_command_buffer_update_capability_flags_t update_capability_flags;
  ASSERT_SUCCESS(urDeviceGetInfo(
      device, UR_DEVICE_INFO_COMMAND_BUFFER_UPDATE_CAPABILITIES_EXP,
      sizeof(update_capability_flags), &update_capability_flags, nullptr));

  if (0 != update_capability_flags) {
    GTEST_SKIP() << "Test requires a device without update support";
  }

  ur_exp_command_buffer_handle_t cmd_buf_handle = nullptr;
  ur_exp_command_buffer_desc_t desc{
      UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC, // stype
      nullptr,                                   // pNext
      true,                                      // isUpdatable
      false,                                     // isInOrder
      false                                      // enableProfiling
  };

  ASSERT_EQ(UR_RESULT_ERROR_UNSUPPORTED_FEATURE,
            urCommandBufferCreateExp(context, device, &desc, &cmd_buf_handle));

  ASSERT_EQ(cmd_buf_handle, nullptr);
};
