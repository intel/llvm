// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>
#include <uur/known_failure.h>

using urQueueReleaseTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urQueueReleaseTest);

TEST_P(urQueueReleaseTest, Success) {
    UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

    ASSERT_SUCCESS(urQueueRetain(queue));

    uint32_t prevRefCount = 0;
    ASSERT_SUCCESS(uur::GetObjectReferenceCount(queue, prevRefCount));

    ASSERT_SUCCESS(urQueueRelease(queue));

    uint32_t refCount = 0;
    ASSERT_SUCCESS(uur::GetObjectReferenceCount(queue, refCount));

    ASSERT_GT(prevRefCount, refCount);
}

TEST_P(urQueueReleaseTest, InvalidNullHandleQueue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urQueueRelease(nullptr));
}
