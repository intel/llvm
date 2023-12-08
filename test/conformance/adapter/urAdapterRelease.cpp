// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

struct urAdapterReleaseTest : uur::runtime::urAdapterTest {
    void SetUp() {
        UUR_RETURN_ON_FATAL_FAILURE(uur::runtime::urAdapterTest::SetUp());
        adapter = adapters[0];
    }

    ur_adapter_handle_t adapter;
};

TEST_F(urAdapterReleaseTest, Success) {
    ASSERT_SUCCESS(urAdapterRetain(adapter));
    EXPECT_SUCCESS(urAdapterRelease(adapter));
}

TEST_F(urAdapterReleaseTest, InvalidNullHandleAdapter) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urAdapterRelease(nullptr));
}
