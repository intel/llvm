// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urPlatformGetTest = uur::urAdapterTest;

UUR_INSTANTIATE_ADAPTER_TEST_SUITE(urPlatformGetTest);

TEST_P(urPlatformGetTest, Success) {
  uint32_t count;
  ASSERT_SUCCESS(urPlatformGet(adapter, 0, nullptr, &count));
  ASSERT_NE(count, 0);
  std::vector<ur_platform_handle_t> platforms(count);
  ASSERT_SUCCESS(urPlatformGet(adapter, count, platforms.data(), nullptr));
  for (auto platform : platforms) {
    ASSERT_NE(nullptr, platform);
  }
}

TEST_P(urPlatformGetTest, InvalidNumEntries) {
  uint32_t count;
  ASSERT_SUCCESS(urPlatformGet(adapter, 0, nullptr, &count));
  std::vector<ur_platform_handle_t> platforms(count);
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urPlatformGet(adapter, 0, platforms.data(), nullptr));
}

TEST_P(urPlatformGetTest, InvalidNullPointer) {
  uint32_t count;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urPlatformGet(nullptr, 0, nullptr, &count));
}

TEST_P(urPlatformGetTest, NullArgs) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_VALUE,
                   urPlatformGet(adapter, 0, nullptr, nullptr));
}
