// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include "uur/raii.h"

using urEventCreateWithNativeHandleTest = uur::event::urEventTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventCreateWithNativeHandleTest);

TEST_P(urEventCreateWithNativeHandleTest, Success) {
    ur_native_handle_t native_event = nullptr;
    if (urEventGetNativeHandle(event, &native_event)) {
        GTEST_SKIP();
    }

    // We cannot assume anything about a native_handle, not even if it's
    // `nullptr` since this could be a valid representation within a backend.
    // We can however convert the native_handle back into a unified-runtime handle
    // and perform some query on it to verify that it works.
    uur::raii::Event evt = nullptr;
    ASSERT_SUCCESS(urEventCreateWithNativeHandle(native_event, context, nullptr,
                                                 evt.ptr()));
    ASSERT_NE(evt, nullptr);

    ur_execution_info_t exec_info;
    ASSERT_SUCCESS(urEventGetInfo(evt, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                                  sizeof(ur_execution_info_t), &exec_info,
                                  nullptr));
}
