// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urPhysicalMemRetainTest = uur::urPhysicalMemTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urPhysicalMemRetainTest);

TEST_P(urPhysicalMemRetainTest, Success) {
    ASSERT_SUCCESS(urPhysicalMemRetain(physical_mem));
    ASSERT_SUCCESS(urPhysicalMemRelease(physical_mem));
}

TEST_P(urPhysicalMemRetainTest, InvalidNullHandlePhysicalMem) {
    ASSERT_EQ_RESULT(urPhysicalMemRetain(nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}
