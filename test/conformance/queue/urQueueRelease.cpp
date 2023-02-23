// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urQueueReleaseTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urQueueReleaseTest);

TEST_P(urQueueReleaseTest, Success) {
    ASSERT_SUCCESS(urQueueRetain(queue));

    const auto prevRefCount = uur::GetObjectReferenceCount(queue);
    ASSERT_TRUE(prevRefCount.has_value());

    ASSERT_SUCCESS(urQueueRelease(queue));

    const auto refCount = uur::GetObjectReferenceCount(queue);
    ASSERT_TRUE(refCount.has_value());

    ASSERT_GT(prevRefCount.value(), refCount.value());
}

TEST_P(urQueueReleaseTest, InvalidNullHandleQueue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urQueueRelease(nullptr));
}
