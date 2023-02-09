// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include "fixtures.h"

using urEventReleaseTest = uur::event::urEventReferenceTest;

/* Check that urEventRelease returns Success */
TEST_P(urEventReleaseTest, Success) { ASSERT_SUCCESS(urEventRelease(event)); }

/* Check that urEventRelease decrements the reference count*/
TEST_P(urEventReleaseTest, CheckReferenceCount) {
    ASSERT_SUCCESS(urEventRetain(event));
    ASSERT_TRUE(checkEventReferenceCount(2));
    ASSERT_SUCCESS(urEventRelease(event));
    ASSERT_TRUE(checkEventReferenceCount(1));
    ASSERT_SUCCESS(urEventRelease(event));
}

using urEventReleaseNegativeTest = uur::urQueueTest;

TEST_P(urEventReleaseNegativeTest, InvalidNullHandle) {
    ASSERT_EQ(urEventRelease(nullptr), UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventReleaseTest);
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventReleaseNegativeTest);
