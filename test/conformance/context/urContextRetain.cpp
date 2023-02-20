// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urContextRetainTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextRetainTest);

TEST_P(urContextRetainTest, Success) {
    const auto prevRefCount = uur::urContextGetReferenceCount(context);
    ASSERT_TRUE(prevRefCount.second);

    ASSERT_SUCCESS(urContextRetain(context));

    const auto refCount = uur::urContextGetReferenceCount(context);
    ASSERT_TRUE(refCount.second);

    ASSERT_EQ(prevRefCount.first + 1, refCount.first);

    EXPECT_SUCCESS(urContextRelease(context));
}

TEST_P(urContextRetainTest, InvalidNullHandleContext) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urContextRetain(nullptr));
}
