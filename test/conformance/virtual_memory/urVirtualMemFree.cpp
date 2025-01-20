// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urVirtualMemFreeTest = uur::urVirtualMemTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urVirtualMemFreeTest);

TEST_P(urVirtualMemFreeTest, Success) {
  ASSERT_SUCCESS(urVirtualMemFree(context, virtual_ptr, size));
  virtual_ptr = nullptr; // set to nullptr to prevent double-free
}

TEST_P(urVirtualMemFreeTest, InvalidNullHandleContext) {
  ASSERT_EQ_RESULT(urVirtualMemFree(nullptr, virtual_ptr, size),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urVirtualMemFreeTest, InvalidNullPointerStart) {
  ASSERT_EQ_RESULT(urVirtualMemFree(context, nullptr, size),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
