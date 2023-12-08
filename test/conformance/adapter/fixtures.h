// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
namespace uur {
namespace runtime {

struct urTest : ::testing::Test {

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

struct urAdapterTest : urTest {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urTest::SetUp());

        uint32_t adapter_count;
        ASSERT_SUCCESS(urAdapterGet(0, nullptr, &adapter_count));
        ASSERT_GT(adapter_count, 0);
        adapters.resize(adapter_count);
        ASSERT_SUCCESS(urAdapterGet(adapter_count, adapters.data(), nullptr));
    }

    void TearDown() override {
        for (auto adapter : adapters) {
            ASSERT_SUCCESS(urAdapterRelease(adapter));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urTest::TearDown());
    }

    std::vector<ur_adapter_handle_t> adapters;
};

} // namespace runtime
} // namespace uur
