// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include "fixtures.h"

using urEventRetainTest = uur::event::urEventReferenceTest;

/* Check that urEventRetain returns Success */
TEST_P(urEventRetainTest, Success) {
    ASSERT_SUCCESS(urEventRetain(event));
    ASSERT_SUCCESS(urEventRelease(event));
    ASSERT_SUCCESS(urEventRelease(event));
}

/* Check that urEventRetain increments the reference count */
TEST_P(urEventRetainTest, CheckReferenceCount) {
    ASSERT_TRUE(checkEventReferenceCount(1));
    ASSERT_SUCCESS(urEventRetain(event));
    ASSERT_TRUE(checkEventReferenceCount(2));
    ASSERT_SUCCESS(urEventRelease(event));
    ASSERT_SUCCESS(urEventRelease(event));
}

using urEventRetainNegativeTest = uur::urQueueTest;

TEST_P(urEventRetainNegativeTest, InvalidNullHandle) {
    ASSERT_EQ(urEventRetain(nullptr), UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventRetainTest);
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventRetainNegativeTest);
