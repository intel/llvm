// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urQueueRetainTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urQueueRetainTest);

TEST_P(urQueueRetainTest, Success) {
    const auto prevRefCount = uur::GetObjectReferenceCount(queue);
    ASSERT_TRUE(prevRefCount.has_value());

    ASSERT_SUCCESS(urQueueRetain(queue));

    const auto refCount = uur::GetObjectReferenceCount(queue);
    ASSERT_TRUE(refCount.has_value());

    ASSERT_LT(prevRefCount.value(), refCount.value());

    EXPECT_SUCCESS(urQueueRelease(queue));
}

TEST_P(urQueueRetainTest, InvalidNullHandleQueue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urQueueRetain(nullptr));
}
