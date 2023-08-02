// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urVirtualMemSetAccessTest = uur::urVirtualMemMappedTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urVirtualMemSetAccessTest);

TEST_P(urVirtualMemSetAccessTest, Success) {
    ASSERT_SUCCESS(urVirtualMemSetAccess(context, virtual_ptr, size,
                                         UR_VIRTUAL_MEM_ACCESS_FLAG_READ_ONLY));

    ur_virtual_mem_access_flags_t flags = 0;
    ASSERT_SUCCESS(urVirtualMemGetInfo(context, virtual_ptr, size,
                                       UR_VIRTUAL_MEM_INFO_ACCESS_MODE,
                                       sizeof(flags), &flags, nullptr));
    ASSERT_TRUE(flags & UR_VIRTUAL_MEM_ACCESS_FLAG_READ_ONLY);
}

TEST_P(urVirtualMemSetAccessTest, InvalidNullHandleContext) {
    ASSERT_EQ_RESULT(
        urVirtualMemSetAccess(nullptr, virtual_ptr, size,
                              UR_VIRTUAL_MEM_ACCESS_FLAG_READ_ONLY),
        UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urVirtualMemSetAccessTest, InvalidNullPointerStart) {
    ASSERT_EQ_RESULT(
        urVirtualMemSetAccess(context, nullptr, size,
                              UR_VIRTUAL_MEM_ACCESS_FLAG_READ_ONLY),
        UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
