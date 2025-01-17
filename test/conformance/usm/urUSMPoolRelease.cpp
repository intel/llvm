// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include <uur/fixtures.h>

using urUSMPoolDestroyTest = uur::urUSMPoolTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urUSMPoolDestroyTest);

TEST_P(urUSMPoolDestroyTest, Success) {
  ASSERT_SUCCESS(urUSMPoolRelease(pool));
  pool = nullptr; // prevent double-delete
}

TEST_P(urUSMPoolDestroyTest, InvalidNullHandleContext) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urUSMPoolRelease(nullptr));
}
