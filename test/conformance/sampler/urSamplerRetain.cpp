// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/known_failure.h>

using urSamplerRetainTest = uur::urSamplerTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urSamplerRetainTest);

TEST_P(urSamplerRetainTest, Success) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  uint32_t prevRefCount = 0;
  ASSERT_SUCCESS(uur::GetObjectReferenceCount(sampler, prevRefCount));

  ASSERT_SUCCESS(urSamplerRetain(sampler));

  uint32_t refCount = 0;
  ASSERT_SUCCESS(uur::GetObjectReferenceCount(sampler, refCount));

  ASSERT_LT(prevRefCount, refCount);

  EXPECT_SUCCESS(urSamplerRelease(sampler));
}

TEST_P(urSamplerRetainTest, InvalidNullHandleSampler) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urSamplerRetain(nullptr));
}
