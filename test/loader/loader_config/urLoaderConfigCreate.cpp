// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include "fixtures.hpp"

struct LoaderConfigCreateTest : ::testing::Test {
  void TearDown() override {
    if (loaderConfig) {
      ASSERT_SUCCESS(urLoaderConfigRelease(loaderConfig));
    }
  }

  ur_loader_config_handle_t loaderConfig = nullptr;
};

TEST_F(LoaderConfigCreateTest, Success) {
  ASSERT_SUCCESS(urLoaderConfigCreate(&loaderConfig));
  ASSERT_TRUE(loaderConfig != nullptr);
}

TEST_F(LoaderConfigCreateTest, InvalidNullPointerLoaderConfig) {
  ASSERT_EQ(UR_RESULT_ERROR_INVALID_NULL_POINTER,
            urLoaderConfigCreate(nullptr));
}
