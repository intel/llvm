// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ur_api.h"
#include <uur/fixtures.h>

using urUSMPoolRetainTest = uur::urUSMPoolTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urUSMPoolRetainTest);

TEST_P(urUSMPoolRetainTest, Success) {
  uint32_t prevRefCount = 0;
  ASSERT_SUCCESS(uur::GetObjectReferenceCount(pool, prevRefCount));

  ASSERT_SUCCESS(urUSMPoolRetain(pool));

  uint32_t refCount = 0;
  ASSERT_SUCCESS(uur::GetObjectReferenceCount(pool, refCount));

  ASSERT_LT(prevRefCount, refCount);

  EXPECT_SUCCESS(urUSMPoolRelease(pool));

  uint32_t afterRefCount = 0;
  ASSERT_SUCCESS(uur::GetObjectReferenceCount(pool, afterRefCount));

  ASSERT_LT(afterRefCount, refCount);
}

TEST_P(urUSMPoolRetainTest, InvalidNullHandlePool) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urUSMPoolRetain(nullptr));
}
