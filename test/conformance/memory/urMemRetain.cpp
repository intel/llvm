// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urMemRetainTest = uur::urMemBufferTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemRetainTest);

TEST_P(urMemRetainTest, Success) {
    ASSERT_SUCCESS(urMemRetain(buffer));
    ASSERT_SUCCESS(urMemRelease(buffer));
}

TEST_P(urMemRetainTest, InvalidNullHandleMem) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE, urMemRetain(nullptr));
}

TEST_P(urMemRetainTest, CheckReferenceCount) {
    uint32_t referenceCount = 0;
    ASSERT_SUCCESS(urMemGetInfo(buffer, UR_MEM_INFO_REFERENCE_COUNT,
                                sizeof(referenceCount), &referenceCount,
                                nullptr));
    ASSERT_EQ(referenceCount, 1);

    { ASSERT_SUCCESS(urMemRetain(buffer)); }

    ASSERT_SUCCESS(urMemGetInfo(buffer, UR_MEM_INFO_REFERENCE_COUNT,
                                sizeof(referenceCount), &referenceCount,
                                nullptr));
    ASSERT_EQ(referenceCount, 2);

    ASSERT_SUCCESS(urMemRelease(buffer));

    ASSERT_SUCCESS(urMemGetInfo(buffer, UR_MEM_INFO_REFERENCE_COUNT,
                                sizeof(referenceCount), &referenceCount,
                                nullptr));
    ASSERT_EQ(referenceCount, 1);
}
