// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urCommandBufferRetainExpTest =
    uur::command_buffer::urCommandBufferExpTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urCommandBufferRetainExpTest);

TEST_P(urCommandBufferRetainExpTest, Success) {
  uint32_t prev_ref_count = 0;
  EXPECT_SUCCESS(uur::GetObjectReferenceCount(cmd_buf_handle, prev_ref_count));

  EXPECT_SUCCESS(urCommandBufferRetainExp(cmd_buf_handle));

  uint32_t ref_count = 0;
  EXPECT_SUCCESS(uur::GetObjectReferenceCount(cmd_buf_handle, ref_count));

  EXPECT_LT(prev_ref_count, ref_count);

  EXPECT_SUCCESS(urCommandBufferReleaseExp(cmd_buf_handle));
}

TEST_P(urCommandBufferRetainExpTest, InvalidNullHandle) {
  EXPECT_EQ_RESULT(urCommandBufferRetainExp(nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}
