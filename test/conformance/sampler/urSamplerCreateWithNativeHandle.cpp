// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urSamplerCreateWithNativeHandleTest = uur::urSamplerTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urSamplerCreateWithNativeHandleTest);

TEST_P(urSamplerCreateWithNativeHandleTest, Success) {
    ur_native_handle_t native_sampler = nullptr;
    if (urSamplerGetNativeHandle(sampler, &native_sampler)) {
        GTEST_SKIP();
    }

    // We cannot assume anything about a native_handle, not even if it's
    // `nullptr` since this could be a valid representation within a backend.
    // We can however convert the native_handle back into a unified-runtime handle
    // and perform some query on it to verify that it works.
    ur_sampler_handle_t hSampler = nullptr;
    ur_sampler_native_properties_t props{};
    ASSERT_SUCCESS(urSamplerCreateWithNativeHandle(native_sampler, context,
                                                   &props, &hSampler));
    ASSERT_NE(hSampler, nullptr);

    ur_sampler_addressing_mode_t addr_mode;
    ASSERT_SUCCESS(urSamplerGetInfo(hSampler, UR_SAMPLER_INFO_ADDRESSING_MODE,
                                    sizeof(addr_mode), &addr_mode, nullptr));
    ASSERT_EQ(addr_mode, sampler_desc.addressingMode);
    ASSERT_SUCCESS(urSamplerRelease(hSampler));
}
