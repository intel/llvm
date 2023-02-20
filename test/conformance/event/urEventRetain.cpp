// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include "fixtures.h"

using urEventRetainTest = uur::event::urEventTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventRetainTest);

TEST_P(urEventRetainTest, Success) {
    const auto prevRefCount = uur::urEventGetReferenceCount(event);
    ASSERT_TRUE(prevRefCount.second);

    ASSERT_SUCCESS(urEventRetain(event));

    const auto refCount = uur::urEventGetReferenceCount(event);
    ASSERT_TRUE(refCount.second);

    ASSERT_LT(prevRefCount.first, refCount.first);

    ASSERT_SUCCESS(urEventRelease(event));
}

TEST_P(urEventRetainTest, InvalidNullHandle) {
    ASSERT_EQ_RESULT(urEventRetain(nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}
