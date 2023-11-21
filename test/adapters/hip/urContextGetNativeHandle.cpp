// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urHipContextGetNativeHandleTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urHipContextGetNativeHandleTest);

TEST_P(urHipContextGetNativeHandleTest, Success) {
    ur_native_handle_t native_context = nullptr;
    ASSERT_SUCCESS(urContextGetNativeHandle(context, &native_context));
    hipCtx_t hip_context = reinterpret_cast<hipCtx_t>(native_context);
    std::ignore = hip_context;
}
