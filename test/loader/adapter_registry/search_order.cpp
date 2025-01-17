// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include "fixtures.hpp"

template <typename P>
void assertRegistryPathSequence(const std::vector<fs::path> &testAdapterPaths,
                                P predicate) {
  static size_t assertIndex = 0;

  auto pathIt = std::find_if(testAdapterPaths.cbegin(), testAdapterPaths.cend(),
                             std::move(predicate));
  size_t index = std::distance(testAdapterPaths.cbegin(), pathIt);
  ASSERT_EQ(index, assertIndex++);
}

TEST_F(adapterRegSearchTest, testSearchOrder) {
  // Adapter search order:
  // 1. Every path from UR_ADAPTERS_SEARCH_PATH.
  // 2. OS search paths (disabled on Windows).
  // 3. Loader library directory.
  auto it = std::find_if(registry.cbegin(), registry.cend(), hasTestLibName);
  ASSERT_NE(it, registry.end());

  const auto &testAdapterPaths = *it;
  assertRegistryPathSequence(testAdapterPaths, isTestEnvPath);
#ifndef _WIN32
  assertRegistryPathSequence(testAdapterPaths, isTestLibName);
#endif
  assertRegistryPathSequence(testAdapterPaths, isCurPath);
}
