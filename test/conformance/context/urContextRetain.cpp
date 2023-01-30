// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urContextRetainTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextRetainTest);

TEST_P(urContextRetainTest, Success) {
    ASSERT_SUCCESS(urContextRetain(context));
    EXPECT_SUCCESS(urContextRelease(context));
}

TEST_P(urContextRetainTest, InvalidNullHandleContext) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urContextRetain(nullptr));
}
