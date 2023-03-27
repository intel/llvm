// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urMemGetNativeHandleTest = uur::urMemBufferTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemGetNativeHandleTest);

TEST_P(urMemGetNativeHandleTest, Success) {
    ur_native_handle_t hNativeMem = nullptr;
    ASSERT_SUCCESS(urMemGetNativeHandle(buffer, &hNativeMem));

    // We cannot assume anything about a native_handle, not even if it's
    // `nullptr` since this could be a valid representation within a backend.
    // We can however convert the native_handle back into a unified-runtime handle
    // and perform some query on it to verify that it works.
    ur_mem_handle_t mem = nullptr;
    ASSERT_SUCCESS(urMemCreateWithNativeHandle(hNativeMem, context, &mem));
    ASSERT_NE(mem, nullptr);

    size_t alloc_size = 0;
    ASSERT_SUCCESS(urMemGetInfo(mem, UR_MEM_INFO_SIZE, sizeof(size_t),
                                &alloc_size, nullptr));

    ASSERT_SUCCESS(urMemRelease(mem));
}

TEST_P(urMemGetNativeHandleTest, InvalidNullHandleMem) {
    ur_native_handle_t phNativeMem;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urMemGetNativeHandle(nullptr, &phNativeMem));
}

TEST_P(urMemGetNativeHandleTest, InvalidNullPointerNativeMem) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urMemGetNativeHandle(buffer, nullptr));
}
