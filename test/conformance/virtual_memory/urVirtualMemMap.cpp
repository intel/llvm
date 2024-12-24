// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urVirtualMemMapWithFlagsTest =
    uur::urVirtualMemTestWithParam<ur_virtual_mem_access_flag_t>;
UUR_TEST_SUITE_P(urVirtualMemMapWithFlagsTest,
                 ::testing::Values(UR_VIRTUAL_MEM_ACCESS_FLAG_NONE,
                                   UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE,
                                   UR_VIRTUAL_MEM_ACCESS_FLAG_READ_ONLY),
                 uur::deviceTestWithParamPrinter<ur_virtual_mem_access_flag_t>);

TEST_P(urVirtualMemMapWithFlagsTest, Success) {
    ASSERT_SUCCESS(urVirtualMemMap(context, virtual_ptr, size, physical_mem, 0,
                                   getParam()));
    EXPECT_SUCCESS(urVirtualMemUnmap(context, virtual_ptr, size));
}

using urVirtualMemMapTest = uur::urVirtualMemTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urVirtualMemMapTest);

TEST_P(urVirtualMemMapTest, InvalidNullHandleContext) {
    ASSERT_EQ_RESULT(urVirtualMemMap(nullptr, virtual_ptr, size, physical_mem,
                                     0, UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urVirtualMemMapTest, InvalidNullHandlePhysicalMem) {
    ASSERT_EQ_RESULT(urVirtualMemMap(context, virtual_ptr, size, nullptr, 0,
                                     UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urVirtualMemMapTest, InvalidNullPointerStart) {
    ASSERT_EQ_RESULT(urVirtualMemMap(context, nullptr, size, physical_mem, 0,
                                     UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urVirtualMemMapTest, InvalidEnumerationFlags) {
    ASSERT_EQ_RESULT(urVirtualMemMap(context, virtual_ptr, size, physical_mem,
                                     0,
                                     UR_VIRTUAL_MEM_ACCESS_FLAG_FORCE_UINT32),
                     UR_RESULT_ERROR_INVALID_ENUMERATION);
}
