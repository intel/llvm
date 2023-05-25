// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urSamplerGetNativeHandleTest = uur::urSamplerTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urSamplerGetNativeHandleTest);

TEST_P(urSamplerGetNativeHandleTest, Success) {
    ur_native_handle_t native_sampler = nullptr;
    ASSERT_SUCCESS(urSamplerGetNativeHandle(sampler, &native_sampler));

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
}

TEST_P(urSamplerGetNativeHandleTest, InvalidNullHandleSampler) {
    ur_native_handle_t native_handle = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urSamplerGetNativeHandle(nullptr, &native_handle));
}

TEST_P(urSamplerGetNativeHandleTest, InvalidNullPointerNativeHandle) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urSamplerGetNativeHandle(sampler, nullptr));
}
