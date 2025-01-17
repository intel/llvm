// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include "fixtures.h"

using urEventReleaseTest = uur::event::urEventTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEventReleaseTest);

TEST_P(urEventReleaseTest, Success) {
  ASSERT_SUCCESS(urEventRetain(event));

  uint32_t prevRefCount = 0;
  ASSERT_SUCCESS(uur::GetObjectReferenceCount(event, prevRefCount));

  ASSERT_SUCCESS(urEventRelease(event));

  uint32_t refCount = 0;
  ASSERT_SUCCESS(uur::GetObjectReferenceCount(event, refCount));

  ASSERT_GT(prevRefCount, refCount);
}

TEST_P(urEventReleaseTest, InvalidNullHandle) {
  ASSERT_EQ_RESULT(urEventRelease(nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}
