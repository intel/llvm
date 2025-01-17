// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#ifndef UR_LOADER_CONFIG_TEST_FIXTURES_H
#define UR_LOADER_CONFIG_TEST_FIXTURES_H

#include "ur_api.h"
#include <algorithm>
#include <gtest/gtest.h>

#ifndef ASSERT_SUCCESS
#define ASSERT_SUCCESS(ACTUAL) ASSERT_EQ(UR_RESULT_SUCCESS, ACTUAL)
#endif

/// @brief Make a string a valid identifier for gtest.
/// @param str The string to sanitize.
inline std::string GTestSanitizeString(const std::string &str) {
  auto str_cpy = str;
  std::replace_if(
      str_cpy.begin(), str_cpy.end(), [](char c) { return !std::isalnum(c); },
      '_');
  return str_cpy;
}

#endif
