// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "fixtures.hpp"

struct urLoaderTearDownTest : testing::Test {
  void SetUp() override {
    ur_device_init_flags_t device_flags = 0;
    ASSERT_SUCCESS(urLoaderInit(device_flags, nullptr));
  }
};

TEST_F(urLoaderTearDownTest, Success) { ASSERT_SUCCESS(urLoaderTearDown()); }
