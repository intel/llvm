// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urVirtualMemUnmapTest = uur::urVirtualMemTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urVirtualMemUnmapTest);

TEST_P(urVirtualMemUnmapTest, Success) {
  ASSERT_SUCCESS(urVirtualMemMap(context, virtual_ptr, size, physical_mem, 0,
                                 UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE));
  ASSERT_SUCCESS(urVirtualMemUnmap(context, virtual_ptr, size));
}

TEST_P(urVirtualMemUnmapTest, InvalidNullHandleContext) {
  ASSERT_EQ_RESULT(urVirtualMemUnmap(nullptr, virtual_ptr, size),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urVirtualMemUnmapTest, InvalidNullPointerStart) {
  ASSERT_EQ_RESULT(urVirtualMemUnmap(context, nullptr, size),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
