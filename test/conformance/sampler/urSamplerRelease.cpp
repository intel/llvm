// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urSamplerReleaseTest = uur::urSamplerTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urSamplerReleaseTest);

TEST_P(urSamplerReleaseTest, Success) {
    ASSERT_SUCCESS(urSamplerRetain(sampler));

    uint32_t prevRefCount = 0;
    ASSERT_SUCCESS(uur::GetObjectReferenceCount(sampler, prevRefCount));

    ASSERT_SUCCESS(urSamplerRelease(sampler));

    uint32_t refCount = 0;
    ASSERT_SUCCESS(uur::GetObjectReferenceCount(sampler, refCount));

    ASSERT_GT(prevRefCount, refCount);
}

TEST_P(urSamplerReleaseTest, InvalidNullHandleSampler) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urSamplerRelease(nullptr));
}
