// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urPlatformGetTest = uur::urAdapterTest;

TEST_F(urPlatformGetTest, Success) {
  uint32_t count;
  ASSERT_SUCCESS(urPlatformGet(adapter, 0, nullptr, &count));
  ASSERT_NE(count, 0);
  std::vector<ur_platform_handle_t> platforms(count);
  ASSERT_SUCCESS(urPlatformGet(adapter, count, platforms.data(), nullptr));
  for (auto platform : platforms) {
    ASSERT_NE(nullptr, platform);
  }
}

TEST_F(urPlatformGetTest, InvalidNumEntries) {
  uint32_t count;
  ASSERT_SUCCESS(urPlatformGet(adapter, 0, nullptr, &count));
  std::vector<ur_platform_handle_t> platforms(count);
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urPlatformGet(adapter, 0, platforms.data(), nullptr));
}

TEST_F(urPlatformGetTest, InvalidNullPointer) {
  uint32_t count;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urPlatformGet(nullptr, 0, nullptr, &count));
}

TEST_F(urPlatformGetTest, NullArgs) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_VALUE,
                   urPlatformGet(adapter, 0, nullptr, nullptr));
}
