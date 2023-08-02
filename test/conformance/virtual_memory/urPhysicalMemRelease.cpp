// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urPhysicalMemReleaseTest = uur::urPhysicalMemTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urPhysicalMemReleaseTest);

TEST_P(urPhysicalMemReleaseTest, Success) {
    ASSERT_SUCCESS(urPhysicalMemRetain(physical_mem));
    ASSERT_SUCCESS(urPhysicalMemRelease(physical_mem));
}

TEST_P(urPhysicalMemReleaseTest, InvalidNullHandlePhysicalMem) {
    ASSERT_EQ_RESULT(urPhysicalMemRelease(nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}
