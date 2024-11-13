// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/raii.h"
#include <uur/fixtures.h>

using urSamplerCreateWithNativeHandleTest = uur::urSamplerTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urSamplerCreateWithNativeHandleTest);

TEST_P(urSamplerCreateWithNativeHandleTest, Success) {
    ur_native_handle_t native_sampler = 0;

    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
        urSamplerGetNativeHandle(sampler, &native_sampler));

    // We cannot assume anything about a native_handle, not even if it's
    // `nullptr` since this could be a valid representation within a backend.
    // We can however convert the native_handle back into a unified-runtime handle
    // and perform some query on it to verify that it works.
    ur_sampler_handle_t hSampler = nullptr;
    ur_sampler_native_properties_t props{};
    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urSamplerCreateWithNativeHandle(
        native_sampler, context, &props, &hSampler));
    ASSERT_NE(hSampler, nullptr);

    ur_sampler_addressing_mode_t addr_mode;
    ASSERT_SUCCESS(urSamplerGetInfo(hSampler, UR_SAMPLER_INFO_ADDRESSING_MODE,
                                    sizeof(addr_mode), &addr_mode, nullptr));
    ASSERT_EQ(addr_mode, sampler_desc.addressingMode);
    ASSERT_SUCCESS(urSamplerRelease(hSampler));
}

TEST_P(urSamplerCreateWithNativeHandleTest, InvalidNullHandle) {
    ur_native_handle_t native_sampler = 0;

    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
        urSamplerGetNativeHandle(sampler, &native_sampler));

    ur_sampler_handle_t hSampler = nullptr;
    ur_sampler_native_properties_t props{};
    ASSERT_EQ(urSamplerCreateWithNativeHandle(native_sampler, nullptr, &props,
                                              &hSampler),
              UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urSamplerCreateWithNativeHandleTest, InvalidNullPointer) {
    ur_native_handle_t native_sampler = 0;

    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
        urSamplerGetNativeHandle(sampler, &native_sampler));

    ur_sampler_native_properties_t props{};
    ASSERT_EQ(urSamplerCreateWithNativeHandle(native_sampler, context, &props,
                                              nullptr),
              UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urSamplerCreateWithNativeHandleTest, SuccessWithOwnedNativeHandle) {

    ur_native_handle_t native_handle = 0;
    uur::raii::Sampler hSampler = nullptr;
    {
        ur_sampler_desc_t sampler_desc{
            UR_STRUCTURE_TYPE_SAMPLER_DESC,  /* stype */
            nullptr,                         /* pNext */
            true,                            /* normalizedCoords */
            UR_SAMPLER_ADDRESSING_MODE_NONE, /* addressing mode */
            UR_SAMPLER_FILTER_MODE_NEAREST,  /* filterMode */
        };

        ASSERT_SUCCESS(urSamplerCreate(context, &sampler_desc, hSampler.ptr()));

        UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
            urSamplerGetNativeHandle(hSampler, &native_handle));
    }

    ur_sampler_native_properties_t props = {
        UR_STRUCTURE_TYPE_SAMPLER_NATIVE_PROPERTIES, nullptr, true};
    ur_sampler_handle_t sampler = nullptr;
    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urSamplerCreateWithNativeHandle(
        native_handle, context, &props, &sampler));
    ASSERT_NE(sampler, nullptr);
}

TEST_P(urSamplerCreateWithNativeHandleTest, SuccessWithUnOwnedNativeHandle) {

    ur_native_handle_t native_handle = 0;
    uur::raii::Sampler hSampler = nullptr;
    {
        ur_sampler_desc_t sampler_desc{
            UR_STRUCTURE_TYPE_SAMPLER_DESC,  /* stype */
            nullptr,                         /* pNext */
            true,                            /* normalizedCoords */
            UR_SAMPLER_ADDRESSING_MODE_NONE, /* addressing mode */
            UR_SAMPLER_FILTER_MODE_NEAREST,  /* filterMode */
        };

        ASSERT_SUCCESS(urSamplerCreate(context, &sampler_desc, hSampler.ptr()));

        UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
            urSamplerGetNativeHandle(hSampler, &native_handle));
    }

    ur_sampler_native_properties_t props = {
        UR_STRUCTURE_TYPE_SAMPLER_NATIVE_PROPERTIES, nullptr, false};
    ur_sampler_handle_t sampler = nullptr;
    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urSamplerCreateWithNativeHandle(
        native_handle, context, &props, &sampler));
    ASSERT_NE(sampler, nullptr);
}
