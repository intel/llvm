// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urAdapterGetLastErrorTest : uur::urAdapterTest {
    int32_t error;
    const char *message = nullptr;
};

UUR_INSTANTIATE_ADAPTER_TEST_SUITE_P(urAdapterGetLastErrorTest);

TEST_P(urAdapterGetLastErrorTest, Success) {
    // We can't reliably generate a UR_RESULT_ERROR_ADAPTER_SPECIFIC error to
    // test the full functionality of this entry point, so instead do a minimal
    // smoke test and check that the call returns successfully, even if no
    // actual error was set.
    ASSERT_EQ_RESULT(UR_RESULT_SUCCESS,
                     urAdapterGetLastError(adapter, &message, &error));
}

TEST_P(urAdapterGetLastErrorTest, InvalidHandle) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urAdapterGetLastError(nullptr, &message, &error));
}

TEST_P(urAdapterGetLastErrorTest, InvalidMessagePtr) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urAdapterGetLastError(adapter, nullptr, &error));
}

TEST_P(urAdapterGetLastErrorTest, InvalidErrorPtr) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urAdapterGetLastError(adapter, &message, nullptr));
}
