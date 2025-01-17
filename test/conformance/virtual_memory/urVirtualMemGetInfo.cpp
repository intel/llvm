// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urVirtualMemGetInfoTest = uur::urVirtualMemMappedTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urVirtualMemGetInfoTest);

TEST_P(urVirtualMemGetInfoTest, SuccessAccessMode) {
    size_t property_size = 0;
    ur_virtual_mem_info_t property_name = UR_VIRTUAL_MEM_INFO_ACCESS_MODE;

    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urVirtualMemGetInfo(context, virtual_ptr, size, property_name, 0,
                            nullptr, &property_size),
        property_name);
    ASSERT_NE(property_size, 0);

    ur_virtual_mem_access_flags_t returned_flags =
        UR_VIRTUAL_MEM_ACCESS_FLAG_FORCE_UINT32;
    ASSERT_SUCCESS(urVirtualMemGetInfo(context, virtual_ptr, size,
                                       property_name, property_size,
                                       &returned_flags, nullptr));

    ASSERT_TRUE(returned_flags & UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE);
}

TEST_P(urVirtualMemGetInfoTest, InvalidNullHandleContext) {
    ur_virtual_mem_access_flags_t flags = 0;
    ASSERT_EQ_RESULT(urVirtualMemGetInfo(nullptr, virtual_ptr, size,
                                         UR_VIRTUAL_MEM_INFO_ACCESS_MODE,
                                         sizeof(flags), &flags, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urVirtualMemGetInfoTest, InvalidNullPointerStart) {
    ur_virtual_mem_access_flags_t flags = 0;
    ASSERT_EQ_RESULT(urVirtualMemGetInfo(context, nullptr, size,
                                         UR_VIRTUAL_MEM_INFO_ACCESS_MODE,
                                         sizeof(flags), &flags, nullptr),
                     UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urVirtualMemGetInfoTest, InvalidEnumerationInfo) {
    size_t property_size = 0;
    ASSERT_EQ_RESULT(urVirtualMemGetInfo(context, virtual_ptr, size,
                                         UR_VIRTUAL_MEM_INFO_FORCE_UINT32, 0,
                                         nullptr, &property_size),
                     UR_RESULT_ERROR_INVALID_ENUMERATION);
}
