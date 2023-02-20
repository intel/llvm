// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urContextReleaseTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextReleaseTest);

TEST_P(urContextReleaseTest, Success) {
    ASSERT_SUCCESS(urContextRetain(context));

    const auto prevRefCount = uur::urContextGetReferenceCount(context);
    ASSERT_TRUE(prevRefCount.second);

    ASSERT_SUCCESS(urContextRelease(context));

    const auto refCount = uur::urContextGetReferenceCount(context);
    ASSERT_TRUE(refCount.second);

    ASSERT_EQ(prevRefCount.first + 1, refCount.first);
}

TEST_P(urContextReleaseTest, InvalidNullHandleContext) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urContextRelease(nullptr));
}
