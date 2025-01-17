// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include "fixtures.hpp"

TEST_F(adapterRegSearchTest, testSearchWithEnv) {
  // Check if there's any path that's just a library name (disabled on Windows).
#ifndef _WIN32
  auto testLibNameExists =
      std::any_of(registry.cbegin(), registry.cend(), hasTestLibName);
  ASSERT_TRUE(testLibNameExists);
#endif

  // Check for path obtained from 'UR_ADAPTERS_SEARCH_PATH'
  auto testEnvPathExists =
      std::any_of(registry.cbegin(), registry.cend(), hasTestEnvPath);
  ASSERT_TRUE(testEnvPathExists);

  // Check for current directory path
  auto testCurPathExists =
      std::any_of(registry.cbegin(), registry.cend(), hasCurPath);
  ASSERT_TRUE(testCurPathExists);
}
