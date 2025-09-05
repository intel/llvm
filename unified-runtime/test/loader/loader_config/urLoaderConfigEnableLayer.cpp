// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.hpp"

struct urLoaderConfigEnableLayerTest : LoaderConfigTest {
  void SetUp() override {
    LoaderConfigTest::SetUp();
    // Get the first available layer to test with
    size_t availableLayersLen = 0;
    ASSERT_SUCCESS(urLoaderConfigGetInfo(loaderConfig,
                                         UR_LOADER_CONFIG_INFO_AVAILABLE_LAYERS,
                                         0, nullptr, &availableLayersLen));
    validLayerName.resize(availableLayersLen);
    ASSERT_SUCCESS(urLoaderConfigGetInfo(
        loaderConfig, UR_LOADER_CONFIG_INFO_AVAILABLE_LAYERS,
        availableLayersLen, validLayerName.data(), nullptr));
    if (validLayerName.find(";") != std::string::npos) {
      validLayerName = validLayerName.substr(0, validLayerName.find(";"));
    }
  }

  std::string validLayerName;
};

TEST_F(urLoaderConfigEnableLayerTest, Success) {
  ASSERT_SUCCESS(
      urLoaderConfigEnableLayer(loaderConfig, validLayerName.data()));
}

TEST_F(urLoaderConfigEnableLayerTest, LayerNotPresent) {
  ASSERT_EQ(UR_RESULT_ERROR_LAYER_NOT_PRESENT,
            urLoaderConfigEnableLayer(loaderConfig, "not a real layer"));
}

TEST_F(urLoaderConfigEnableLayerTest, InvalidNullHandleLoaderConfig) {
  ASSERT_EQ(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
            urLoaderConfigEnableLayer(nullptr, validLayerName.data()));
}

TEST_F(urLoaderConfigEnableLayerTest, InvalidNullPointerLayerName) {
  ASSERT_EQ(UR_RESULT_ERROR_INVALID_NULL_POINTER,
            urLoaderConfigEnableLayer(loaderConfig, nullptr));
}
