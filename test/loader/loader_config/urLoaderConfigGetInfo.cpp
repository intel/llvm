// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.hpp"

struct urLoaderConfigGetInfoTest
    : LoaderConfigTest,
      ::testing::WithParamInterface<ur_loader_config_info_t> {
    void SetUp() override {
        LoaderConfigTest::SetUp();
        infoType = GetParam();
        ASSERT_SUCCESS(urLoaderConfigGetInfo(loaderConfig, infoType, 0, nullptr,
                                             &infoSize));
        EXPECT_NE(0, infoSize);
        infoAllocation.resize(infoSize);
    }

    ur_loader_config_info_t infoType;
    size_t infoSize = 0;
    std::vector<char> infoAllocation;
};

INSTANTIATE_TEST_SUITE_P(
    , urLoaderConfigGetInfoTest,
    ::testing::Values(UR_LOADER_CONFIG_INFO_AVAILABLE_LAYERS,
                      UR_LOADER_CONFIG_INFO_REFERENCE_COUNT));

TEST_P(urLoaderConfigGetInfoTest, Success) {
    ASSERT_SUCCESS(urLoaderConfigGetInfo(loaderConfig, infoType, infoSize,
                                         infoAllocation.data(), nullptr));
}

TEST_P(urLoaderConfigGetInfoTest, InvalidNullHandleLoaderConfig) {
    ASSERT_EQ(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
              urLoaderConfigGetInfo(nullptr, infoType, infoSize,
                                    infoAllocation.data(), nullptr));
}

TEST_P(urLoaderConfigGetInfoTest, InvalidNullPointer) {
    ASSERT_EQ(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urLoaderConfigGetInfo(loaderConfig, infoType, 0, nullptr, nullptr));
}

TEST_P(urLoaderConfigGetInfoTest, InvalidEnumerationInfoType) {
    ASSERT_EQ(UR_RESULT_ERROR_INVALID_ENUMERATION,
              urLoaderConfigGetInfo(loaderConfig,
                                    UR_LOADER_CONFIG_INFO_FORCE_UINT32, 0,
                                    nullptr, &infoSize));
}
