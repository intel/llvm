// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urAdapterGetTest : ::testing::Test {
    void SetUp() override {
        ur_device_init_flags_t device_flags = 0;
        ASSERT_SUCCESS(urLoaderConfigCreate(&loader_config));
        ASSERT_SUCCESS(urLoaderConfigEnableLayer(loader_config,
                                                 "UR_LAYER_FULL_VALIDATION"));
        ASSERT_SUCCESS(urLoaderInit(device_flags, loader_config));
    }

    void TearDown() override {
        if (loader_config) {
            ASSERT_SUCCESS(urLoaderConfigRelease(loader_config));
        }
        ASSERT_SUCCESS(urLoaderTearDown());
    }

    ur_loader_config_handle_t loader_config = nullptr;
};

TEST_F(urAdapterGetTest, Success) {
    uint32_t adapter_count;
    ASSERT_SUCCESS(urAdapterGet(0, nullptr, &adapter_count));
    std::vector<ur_adapter_handle_t> adapters(adapter_count);
    ASSERT_SUCCESS(urAdapterGet(adapter_count, adapters.data(), nullptr));
}

TEST_F(urAdapterGetTest, InvalidNumEntries) {
    uint32_t adapter_count;
    ASSERT_SUCCESS(urAdapterGet(0, nullptr, &adapter_count));
    std::vector<ur_adapter_handle_t> adapters(adapter_count);
    ASSERT_EQ(urAdapterGet(0, adapters.data(), nullptr),
              UR_RESULT_ERROR_INVALID_SIZE);
}
