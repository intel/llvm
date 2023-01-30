// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urMemCreateWithNativeHandleTest = uur::urMemBufferTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemCreateWithNativeHandleTest);

// TODO - implement valid native handle test - #178
TEST_P(urMemCreateWithNativeHandleTest, DISABLED_Success) {
    ur_native_handle_t native_handle = nullptr;
    ur_mem_handle_t mem = nullptr;
    ASSERT_SUCCESS(urMemCreateWithNativeHandle(native_handle, context, &mem));
}
