// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include <uur/fixtures.h>

using urContextRetainTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urContextRetainTest);

TEST_P(urContextRetainTest, Success) {
  uint32_t prevRefCount = 0;
  ASSERT_SUCCESS(uur::GetObjectReferenceCount(context, prevRefCount));

  ASSERT_SUCCESS(urContextRetain(context));

  uint32_t refCount = 0;
  ASSERT_SUCCESS(uur::GetObjectReferenceCount(context, refCount));

  ASSERT_LT(prevRefCount, refCount);

  EXPECT_SUCCESS(urContextRelease(context));
}

TEST_P(urContextRetainTest, InvalidNullHandleContext) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urContextRetain(nullptr));
}
