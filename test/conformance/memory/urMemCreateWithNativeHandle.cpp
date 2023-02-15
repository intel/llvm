// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urMemCreateWithNativeHandleTest = uur::urMemBufferTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemCreateWithNativeHandleTest);

TEST_P(urMemCreateWithNativeHandleTest, InvalidNullHandleNativeMem) {
    ur_mem_handle_t mem = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urMemCreateWithNativeHandle(nullptr, context, &mem));
}
