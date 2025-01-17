// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include "fixtures.hpp"

using urLoaderConfigRetainTest = LoaderConfigTest;

TEST_F(urLoaderConfigRetainTest, Success) {
  uint32_t prevRefCount = 0;
  ASSERT_SUCCESS(
      urLoaderConfigGetInfo(loaderConfig, UR_LOADER_CONFIG_INFO_REFERENCE_COUNT,
                            sizeof(prevRefCount), &prevRefCount, nullptr));

  ASSERT_SUCCESS(urLoaderConfigRetain(loaderConfig));

  uint32_t refCount = 0;
  ASSERT_SUCCESS(urLoaderConfigGetInfo(loaderConfig,
                                       UR_LOADER_CONFIG_INFO_REFERENCE_COUNT,
                                       sizeof(refCount), &refCount, nullptr));

  ASSERT_GT(refCount, prevRefCount);

  ASSERT_SUCCESS(urLoaderConfigRelease(loaderConfig));
}

TEST_F(urLoaderConfigRetainTest, InvalidNullHandleLoaderConfig) {
  ASSERT_EQ(UR_RESULT_ERROR_INVALID_NULL_HANDLE, urLoaderConfigRetain(nullptr));
}
