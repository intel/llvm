// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using CommandBufferNativeTest = uur::command_buffer::urCommandBufferExpTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(CommandBufferNativeTest);

TEST_P(CommandBufferNativeTest, GetNativeHandle) {
  ur_native_handle_t native_cmd_buf{};
  ASSERT_SUCCESS(
      urCommandBufferGetNativeHandleExp(cmd_buf_handle, &native_cmd_buf));
  ASSERT_NE(ur_native_handle_t{}, native_cmd_buf);
}
