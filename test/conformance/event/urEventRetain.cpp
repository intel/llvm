// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include "fixtures.h"

using urEventRetainTest = uur::event::urEventTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventRetainTest);

TEST_P(urEventRetainTest, Success) {
    const auto prevRefCount = uur::GetObjectReferenceCount(event);
    ASSERT_TRUE(prevRefCount.has_value());

    ASSERT_SUCCESS(urEventRetain(event));

    const auto refCount = uur::GetObjectReferenceCount(event);
    ASSERT_TRUE(refCount.has_value());

    ASSERT_LT(prevRefCount.value(), refCount.value());

    ASSERT_SUCCESS(urEventRelease(event));
}

TEST_P(urEventRetainTest, InvalidNullHandle) {
    ASSERT_EQ_RESULT(urEventRetain(nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}
