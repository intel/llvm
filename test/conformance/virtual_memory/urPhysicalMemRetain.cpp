// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urPhysicalMemRetainTest = uur::urPhysicalMemTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urPhysicalMemRetainTest);

TEST_P(urPhysicalMemRetainTest, Success) {
    uint32_t referenceCount = 0;
    ASSERT_SUCCESS(
        urPhysicalMemGetInfo(physical_mem, UR_PHYSICAL_MEM_INFO_REFERENCE_COUNT,
                             sizeof(referenceCount), &referenceCount, nullptr));
    ASSERT_GE(referenceCount, 1);

    ASSERT_SUCCESS(urPhysicalMemRetain(physical_mem));
    ASSERT_SUCCESS(
        urPhysicalMemGetInfo(physical_mem, UR_PHYSICAL_MEM_INFO_REFERENCE_COUNT,
                             sizeof(referenceCount), &referenceCount, nullptr));
    ASSERT_EQ(referenceCount, 2);

    ASSERT_SUCCESS(urPhysicalMemRelease(physical_mem));
    ASSERT_SUCCESS(
        urPhysicalMemGetInfo(physical_mem, UR_PHYSICAL_MEM_INFO_REFERENCE_COUNT,
                             sizeof(referenceCount), &referenceCount, nullptr));
    ASSERT_EQ(referenceCount, 1);
}

TEST_P(urPhysicalMemRetainTest, InvalidNullHandlePhysicalMem) {
    ASSERT_EQ_RESULT(urPhysicalMemRetain(nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}
