// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urAdapterGetTest = uur::runtime::urTest;

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
    ASSERT_SUCCESS(urAdapterGet(0, adapters.data(), nullptr));
}
