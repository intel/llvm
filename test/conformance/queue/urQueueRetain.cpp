// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>
#include <uur/known_failure.h>

using urQueueRetainTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urQueueRetainTest);

TEST_P(urQueueRetainTest, Success) {
    UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

    uint32_t prevRefCount = 0;
    ASSERT_SUCCESS(uur::GetObjectReferenceCount(queue, prevRefCount));

    ASSERT_SUCCESS(urQueueRetain(queue));

    uint32_t refCount = 0;
    ASSERT_SUCCESS(uur::GetObjectReferenceCount(queue, refCount));

    ASSERT_LT(prevRefCount, refCount);

    EXPECT_SUCCESS(urQueueRelease(queue));
}

TEST_P(urQueueRetainTest, InvalidNullHandleQueue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urQueueRetain(nullptr));
}
