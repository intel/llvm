// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urSamplerGetNativeHandleTest = uur::urSamplerTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urSamplerGetNativeHandleTest);

TEST_P(urSamplerGetNativeHandleTest, Success) {
    ur_native_handle_t native_sampler = nullptr;
    ASSERT_SUCCESS(urSamplerGetNativeHandle(sampler, &native_sampler));
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
