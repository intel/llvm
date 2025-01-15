// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urProgramReleaseTest = uur::urProgramTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urProgramReleaseTest);

TEST_P(urProgramReleaseTest, Success) {
  ASSERT_SUCCESS(urProgramRetain(program));
  ASSERT_SUCCESS(urProgramRelease(program));
}

TEST_P(urProgramReleaseTest, InvalidNullHandleProgram) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urProgramRelease(nullptr));
}
