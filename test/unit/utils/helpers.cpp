// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ur_util.hpp"

TEST(groupDigits, Success) {
  EXPECT_EQ(groupDigits(-1), "-1");
  EXPECT_EQ(groupDigits(-12), "-12");
  EXPECT_EQ(groupDigits(-123), "-123");
  EXPECT_EQ(groupDigits(-1234), "-1'234");
  EXPECT_EQ(groupDigits(-12345), "-12'345");
  EXPECT_EQ(groupDigits(-123456), "-123'456");
  EXPECT_EQ(groupDigits(-1234567), "-1'234'567");
  EXPECT_EQ(groupDigits(-12345678), "-12'345'678");

  EXPECT_EQ(groupDigits(0), "0");
  EXPECT_EQ(groupDigits(1), "1");
  EXPECT_EQ(groupDigits(12), "12");
  EXPECT_EQ(groupDigits(123), "123");
  EXPECT_EQ(groupDigits(1234), "1'234");
  EXPECT_EQ(groupDigits(12345), "12'345");
  EXPECT_EQ(groupDigits(123456), "123'456");
  EXPECT_EQ(groupDigits(1234567), "1'234'567");
  EXPECT_EQ(groupDigits(12345678), "12'345'678");
}
