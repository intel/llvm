// Copyright (C) 2023-2026 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urProgramRetainTest = uur::urProgramTest;
UUR_DEVICE_TEST_SUITE_WITH_DEFAULT_QUEUE(urProgramRetainTest);

TEST_P(urProgramRetainTest, Success) {
  uint32_t prevRefCount = 0;
  ASSERT_SUCCESS(uur::GetObjectReferenceCount(program, prevRefCount));

  ASSERT_SUCCESS(urProgramRetain(program));

  uint32_t refCount = 0;
  ASSERT_SUCCESS(uur::GetObjectReferenceCount(program, refCount));

  ASSERT_LT(prevRefCount, refCount);

  ASSERT_SUCCESS(urProgramRetain(program));
}

TEST_P(urProgramRetainTest, InvalidNullHandleProgram) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urProgramRetain(nullptr));
}
