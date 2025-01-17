// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#ifndef UR_LOADER_CONFIG_TEST_FIXTURES_H
#define UR_LOADER_CONFIG_TEST_FIXTURES_H

#include "ur_api.h"
#include <gtest/gtest.h>

#ifndef ASSERT_SUCCESS
#define ASSERT_SUCCESS(ACTUAL) ASSERT_EQ(UR_RESULT_SUCCESS, ACTUAL)
#endif

struct LoaderConfigTest : ::testing::Test {
  void SetUp() override { ASSERT_SUCCESS(urLoaderConfigCreate(&loaderConfig)); }

  void TearDown() override {
    if (loaderConfig) {
      ASSERT_SUCCESS(urLoaderConfigRelease(loaderConfig));
    }
  }

  ur_loader_config_handle_t loaderConfig = nullptr;
};

#endif
