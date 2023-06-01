// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urMemBufferCreateWithNativeHandleTest = uur::urMemBufferTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemBufferCreateWithNativeHandleTest);

TEST_P(urMemBufferCreateWithNativeHandleTest, InvalidNullHandleNativeMem) {
    ur_mem_handle_t mem = nullptr;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urMemBufferCreateWithNativeHandle(nullptr, context, nullptr, &mem));
}
