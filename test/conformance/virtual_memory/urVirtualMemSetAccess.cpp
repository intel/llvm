// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urVirtualMemSetAccessWithFlagsTest =
    uur::urVirtualMemMappedTestWithParam<ur_virtual_mem_access_flag_t>;
UUR_TEST_SUITE_P(urVirtualMemSetAccessWithFlagsTest,
                 ::testing::Values(UR_VIRTUAL_MEM_ACCESS_FLAG_NONE,
                                   UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE,
                                   UR_VIRTUAL_MEM_ACCESS_FLAG_READ_ONLY),
                 uur::deviceTestWithParamPrinter<ur_virtual_mem_access_flag_t>);

TEST_P(urVirtualMemSetAccessWithFlagsTest, Success) {
    ASSERT_SUCCESS(
        urVirtualMemSetAccess(context, virtual_ptr, size, getParam()));

    ur_virtual_mem_access_flags_t flags = 0;
    ASSERT_SUCCESS(urVirtualMemGetInfo(context, virtual_ptr, size,
                                       UR_VIRTUAL_MEM_INFO_ACCESS_MODE,
                                       sizeof(flags), &flags, nullptr));
    if (getParam() == UR_VIRTUAL_MEM_ACCESS_FLAG_NONE) {
        ASSERT_TRUE(flags == 0 || flags == UR_VIRTUAL_MEM_ACCESS_FLAG_NONE);
    } else {
        ASSERT_TRUE(flags & getParam());
    }
}

using urVirtualMemSetAccessTest = uur::urVirtualMemMappedTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urVirtualMemSetAccessTest);

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

TEST_P(urVirtualMemSetAccessTest, InvalidEnumeration) {
    ASSERT_EQ_RESULT(
        urVirtualMemSetAccess(context, virtual_ptr, size,
                              UR_VIRTUAL_MEM_ACCESS_FLAG_FORCE_UINT32),
        UR_RESULT_ERROR_INVALID_ENUMERATION);
}
