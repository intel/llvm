// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception
#include "fixtures.hpp"

struct urLoaderTearDownTest : testing::Test {
  void SetUp() override {
    ur_device_init_flags_t device_flags = 0;
    ASSERT_SUCCESS(urLoaderInit(device_flags, nullptr));
  }
};

TEST_F(urLoaderTearDownTest, Success) { ASSERT_SUCCESS(urLoaderTearDown()); }
