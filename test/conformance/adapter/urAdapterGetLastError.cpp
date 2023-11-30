// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

struct urAdapterGetLastErrorTest : uur::runtime::urAdapterTest {
    int32_t error;
    const char *message = nullptr;
};

TEST_F(urAdapterGetLastErrorTest, Success) {
    // We can't reliably generate a UR_RESULT_ERROR_ADAPTER_SPECIFIC error to
    // test the full functionality of this entry point, so instead do a minimal
    // smoke test and check that the call returns successfully, even if no
    // actual error was set.
    ASSERT_EQ_RESULT(UR_RESULT_SUCCESS,
                     urAdapterGetLastError(adapters[0], &message, &error));
}

TEST_F(urAdapterGetLastErrorTest, InvalidHandle) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urAdapterGetLastError(nullptr, &message, &error));
}

TEST_F(urAdapterGetLastErrorTest, InvalidMessagePtr) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urAdapterGetLastError(adapters[0], nullptr, &error));
}

TEST_F(urAdapterGetLastErrorTest, InvalidErrorPtr) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urAdapterGetLastError(adapters[0], &message, nullptr));
}
