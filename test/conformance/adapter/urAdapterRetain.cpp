// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urAdapterRetainTest = uur::urAdapterTest;
UUR_INSTANTIATE_ADAPTER_TEST_SUITE_P(urAdapterRetainTest);

TEST_P(urAdapterRetainTest, Success) {
    uint32_t referenceCountBefore = 0;

    ASSERT_SUCCESS(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_REFERENCE_COUNT,
                                    sizeof(referenceCountBefore),
                                    &referenceCountBefore, nullptr));

    uint32_t referenceCountAfter = 0;
    EXPECT_SUCCESS(urAdapterRetain(adapter));
    ASSERT_SUCCESS(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_REFERENCE_COUNT,
                                    sizeof(referenceCountAfter),
                                    &referenceCountAfter, nullptr));

    ASSERT_GT(referenceCountAfter, referenceCountBefore);
}

TEST_P(urAdapterRetainTest, InvalidNullHandleAdapter) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urAdapterRetain(nullptr));
}
