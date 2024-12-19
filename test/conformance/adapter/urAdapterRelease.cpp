// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urAdapterReleaseTest = uur::urAdapterTest;
UUR_INSTANTIATE_ADAPTER_TEST_SUITE_P(urAdapterReleaseTest);

TEST_P(urAdapterReleaseTest, Success) {
    uint32_t referenceCountBefore = 0;
    ASSERT_SUCCESS(urAdapterRetain(adapter));

    ASSERT_SUCCESS(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_REFERENCE_COUNT,
                                    sizeof(referenceCountBefore),
                                    &referenceCountBefore, nullptr));

    uint32_t referenceCountAfter = 0;
    EXPECT_SUCCESS(urAdapterRelease(adapter));
    ASSERT_SUCCESS(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_REFERENCE_COUNT,
                                    sizeof(referenceCountAfter),
                                    &referenceCountAfter, nullptr));

    ASSERT_LE(referenceCountAfter, referenceCountBefore);
}

TEST_P(urAdapterReleaseTest, InvalidNullHandleAdapter) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urAdapterRelease(nullptr));
}
