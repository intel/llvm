// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include "fixtures.hpp"

using urLoaderConfigReleaseTest = LoaderConfigTest;

TEST_F(urLoaderConfigReleaseTest, Success) {
  ASSERT_SUCCESS(urLoaderConfigRetain(loaderConfig));

  uint32_t prevRefCount = 0;
  ASSERT_SUCCESS(
      urLoaderConfigGetInfo(loaderConfig, UR_LOADER_CONFIG_INFO_REFERENCE_COUNT,
                            sizeof(prevRefCount), &prevRefCount, nullptr));

  ASSERT_SUCCESS(urLoaderConfigRelease(loaderConfig));

  uint32_t refCount = 0;
  ASSERT_SUCCESS(urLoaderConfigGetInfo(loaderConfig,
                                       UR_LOADER_CONFIG_INFO_REFERENCE_COUNT,
                                       sizeof(refCount), &refCount, nullptr));

  ASSERT_LT(refCount, prevRefCount);
}

TEST_F(urLoaderConfigReleaseTest, InvalidNullHandleLoaderConfig) {
  ASSERT_EQ(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
            urLoaderConfigRelease(nullptr));
}
