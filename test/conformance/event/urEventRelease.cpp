// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include "fixtures.h"

using urEventReleaseTest = uur::event::urEventTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventReleaseTest);

TEST_P(urEventReleaseTest, Success) {
    ASSERT_SUCCESS(urEventRetain(event));

    const auto prevRefCount = uur::GetObjectReferenceCount(event);
    ASSERT_TRUE(prevRefCount.has_value());

    ASSERT_SUCCESS(urEventRelease(event));

    const auto refCount = uur::GetObjectReferenceCount(event);
    ASSERT_TRUE(refCount.has_value());

    ASSERT_GT(prevRefCount.value(), refCount.value());
}

TEST_P(urEventReleaseTest, InvalidNullHandle) {
    ASSERT_EQ_RESULT(urEventRelease(nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}
