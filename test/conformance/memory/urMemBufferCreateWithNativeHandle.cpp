// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urMemBufferCreateWithNativeHandleTest = uur::urMemBufferTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemBufferCreateWithNativeHandleTest);

TEST_P(urMemBufferCreateWithNativeHandleTest, Success) {
    ur_native_handle_t hNativeMem = nullptr;
    if (urMemGetNativeHandle(buffer, device, &hNativeMem)) {
        GTEST_SKIP();
    }

    // We cannot assume anything about a native_handle, not even if it's
    // `nullptr` since this could be a valid representation within a backend.
    // We can however convert the native_handle back into a unified-runtime handle
    // and perform some query on it to verify that it works.
    ur_mem_handle_t mem = nullptr;
    ur_mem_native_properties_t props = {
        /*.stype =*/UR_STRUCTURE_TYPE_MEM_NATIVE_PROPERTIES,
        /*.pNext =*/nullptr,
        /*.isNativeHandleOwned =*/false,
    };
    ASSERT_SUCCESS(
        urMemBufferCreateWithNativeHandle(hNativeMem, context, &props, &mem));
    ASSERT_NE(mem, nullptr);

    size_t alloc_size = 0;
    ASSERT_SUCCESS(urMemGetInfo(mem, UR_MEM_INFO_SIZE, sizeof(size_t),
                                &alloc_size, nullptr));

    ASSERT_SUCCESS(urMemRelease(mem));
}
