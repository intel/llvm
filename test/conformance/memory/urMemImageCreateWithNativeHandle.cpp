// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urMemImageCreateWithNativeHandleTest = uur::urMemImageTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemImageCreateWithNativeHandleTest);

TEST_P(urMemImageCreateWithNativeHandleTest, Success) {
    ur_native_handle_t native_handle = nullptr;
    if (urMemGetNativeHandle(image, device, &native_handle)) {
        GTEST_SKIP();
    }

    ur_mem_handle_t mem = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urMemImageCreateWithNativeHandle(native_handle, context, &image_format,
                                         &image_desc, nullptr, &mem));
    ASSERT_NE(nullptr, mem);

    ur_context_handle_t mem_context = nullptr;
    ASSERT_SUCCESS(urMemGetInfo(mem, UR_MEM_INFO_CONTEXT,
                                sizeof(ur_context_handle_t), &mem_context,
                                nullptr));
    ASSERT_EQ(context, mem_context);
}
