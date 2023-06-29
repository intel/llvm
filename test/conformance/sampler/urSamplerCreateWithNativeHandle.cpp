// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urSamplerCreateWithNativeHandleTest = uur::urContextTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urSamplerCreateWithNativeHandleTest);

TEST_P(urSamplerCreateWithNativeHandleTest, InvalidNullHandleNativeHandle) {
    ur_sampler_handle_t sampler = nullptr;
    ur_sampler_native_properties_t props{};
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urSamplerCreateWithNativeHandle(nullptr, context, &props, &sampler));
}
