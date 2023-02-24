// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urContextReleaseTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextReleaseTest);

TEST_P(urContextReleaseTest, Success) {
    ASSERT_SUCCESS(urContextRetain(context));

    const auto prevRefCount = uur::GetObjectReferenceCount(context);
    ASSERT_TRUE(prevRefCount.has_value());

    ASSERT_SUCCESS(urContextRelease(context));

    const auto refCount = uur::GetObjectReferenceCount(context);
    ASSERT_TRUE(refCount.has_value());

    ASSERT_GT(prevRefCount.value(), refCount.value());
}

TEST_P(urContextReleaseTest, InvalidNullHandleContext) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urContextRelease(nullptr));
}
