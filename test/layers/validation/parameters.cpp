// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include "fixtures.hpp"

TEST_F(valPlatformsTest, testUrPlatformGetApiVersion) {
  ur_api_version_t api_version = {};

  ASSERT_EQ(urPlatformGetApiVersion(nullptr, &api_version),
            UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  for (auto p : platforms) {
    ASSERT_EQ(urPlatformGetApiVersion(p, nullptr),
              UR_RESULT_ERROR_INVALID_NULL_POINTER);
  }
}
