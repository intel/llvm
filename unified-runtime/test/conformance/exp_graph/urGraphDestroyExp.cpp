// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urGraphDestroyExpTest = uur::urGraphSupportedExpTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urGraphDestroyExpTest);

/* TODO: Test destroying graph with active executable graph instances. */

TEST_P(urGraphDestroyExpTest, InvalidNullHandle) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urGraphDestroyExp(nullptr));
}
