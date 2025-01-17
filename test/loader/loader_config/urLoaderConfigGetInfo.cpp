// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include "fixtures.hpp"

#include <algorithm>

struct urLoaderConfigGetInfoWithParamTest
    : LoaderConfigTest,
      ::testing::WithParamInterface<ur_loader_config_info_t> {
  void SetUp() override {
    LoaderConfigTest::SetUp();
    infoType = GetParam();
    ASSERT_SUCCESS(
        urLoaderConfigGetInfo(loaderConfig, infoType, 0, nullptr, &infoSize));
    EXPECT_NE(0, infoSize);
    infoAllocation.resize(infoSize);
  }

  ur_loader_config_info_t infoType;
  size_t infoSize = 0;
  std::vector<char> infoAllocation;
};

INSTANTIATE_TEST_SUITE_P(
    , urLoaderConfigGetInfoWithParamTest,
    ::testing::Values(UR_LOADER_CONFIG_INFO_AVAILABLE_LAYERS,
                      UR_LOADER_CONFIG_INFO_REFERENCE_COUNT));

TEST_P(urLoaderConfigGetInfoWithParamTest, Success) {
  ASSERT_SUCCESS(urLoaderConfigGetInfo(loaderConfig, infoType, infoSize,
                                       infoAllocation.data(), nullptr));
}

TEST_P(urLoaderConfigGetInfoWithParamTest, InvalidNullHandleLoaderConfig) {
  ASSERT_EQ(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
            urLoaderConfigGetInfo(nullptr, infoType, infoSize,
                                  infoAllocation.data(), nullptr));
}

TEST_P(urLoaderConfigGetInfoWithParamTest, InvalidNullPointer) {
  ASSERT_EQ(UR_RESULT_ERROR_INVALID_NULL_POINTER,
            urLoaderConfigGetInfo(loaderConfig, infoType, 1, nullptr, nullptr));

  ASSERT_EQ(UR_RESULT_ERROR_INVALID_NULL_POINTER,
            urLoaderConfigGetInfo(loaderConfig, infoType, 0, nullptr, nullptr));
}

TEST_P(urLoaderConfigGetInfoWithParamTest, InvalidEnumerationInfoType) {
  ASSERT_EQ(UR_RESULT_ERROR_INVALID_ENUMERATION,
            urLoaderConfigGetInfo(loaderConfig,
                                  UR_LOADER_CONFIG_INFO_FORCE_UINT32, 0,
                                  nullptr, &infoSize));
}

TEST_P(urLoaderConfigGetInfoWithParamTest, InvalidSize) {
  ASSERT_EQ(UR_RESULT_ERROR_INVALID_SIZE,
            urLoaderConfigGetInfo(loaderConfig, infoType, 0,
                                  infoAllocation.data(), &infoSize));

  ASSERT_EQ(UR_RESULT_ERROR_INVALID_SIZE,
            urLoaderConfigGetInfo(loaderConfig, infoType, infoSize - 1,
                                  infoAllocation.data(), &infoSize));
}

using urLoaderConfigGetInfoTest = LoaderConfigTest;

TEST_F(urLoaderConfigGetInfoTest, ReferenceCountNonZero) {
  uint32_t referenceCount = 0;
  ASSERT_SUCCESS(
      urLoaderConfigGetInfo(loaderConfig, UR_LOADER_CONFIG_INFO_REFERENCE_COUNT,
                            sizeof(referenceCount), &referenceCount, nullptr));
  ASSERT_GT(referenceCount, 0);
}

std::vector<std::string> splitString(const std::string &str, char delimiter) {
  std::vector<std::string> tokens;
  std::stringstream ss(str);
  std::string token;
  while (std::getline(ss, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

bool isLayerStringValid(std::string &layersString,
                        const std::vector<std::string> &validLayers) {
  if (layersString.empty()) {
    return true;
  }

  layersString.pop_back(); // remove null terminator before comparing
  std::vector<std::string> layers = splitString(layersString, ';');

  for (const std::string &layer : layers) {
    if (std::find(validLayers.begin(), validLayers.end(), layer) ==
        validLayers.end()) {
      return false;
    }
  }

  return true;
}

TEST_F(urLoaderConfigGetInfoTest, ValidLayersList) {
  std::vector<std::string> layerNames{
      "UR_LAYER_PARAMETER_VALIDATION",
      "UR_LAYER_BOUNDS_CHECKING",
      "UR_LAYER_LEAK_CHECKING",
      "UR_LAYER_LIFETIME_VALIDATION",
      "UR_LAYER_FULL_VALIDATION",
      "UR_LAYER_TRACING",
      "UR_LAYER_ASAN",
      "UR_LAYER_MSAN",
      "UR_LAYER_TSAN",
  };

  std::string availableLayers;
  size_t availableLayersLength = 0;

  ASSERT_SUCCESS(urLoaderConfigGetInfo(loaderConfig,
                                       UR_LOADER_CONFIG_INFO_AVAILABLE_LAYERS,
                                       0, nullptr, &availableLayersLength));

  availableLayers.resize(availableLayersLength);
  ASSERT_SUCCESS(urLoaderConfigGetInfo(
      loaderConfig, UR_LOADER_CONFIG_INFO_AVAILABLE_LAYERS,
      availableLayersLength, availableLayers.data(), nullptr));

  ASSERT_TRUE(isLayerStringValid(availableLayers, layerNames));
}
