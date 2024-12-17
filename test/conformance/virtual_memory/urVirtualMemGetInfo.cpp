// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urVirtualMemGetInfoTestWithParam =
    uur::urVirtualMemMappedTestWithParam<ur_virtual_mem_info_t>;
UUR_TEST_SUITE_P(urVirtualMemGetInfoTestWithParam,
                 ::testing::Values(UR_VIRTUAL_MEM_INFO_ACCESS_MODE),
                 uur::deviceTestWithParamPrinter<ur_virtual_mem_info_t>);

TEST_P(urVirtualMemGetInfoTestWithParam, Success) {
    size_t info_size = 0;
    ur_virtual_mem_info_t info = getParam();
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(urVirtualMemGetInfo(context, virtual_ptr,
                                                         size, info, 0, nullptr,
                                                         &info_size),
                                     info);
    ASSERT_NE(info_size, 0);

    std::vector<uint8_t> data(info_size);
    ASSERT_SUCCESS(urVirtualMemGetInfo(context, virtual_ptr, size, info,
                                       data.size(), data.data(), nullptr));

    switch (info) {
    case UR_VIRTUAL_MEM_INFO_ACCESS_MODE: {
        ASSERT_EQ(sizeof(ur_virtual_mem_access_flags_t), data.size());
        ur_virtual_mem_access_flags_t flags =
            *reinterpret_cast<ur_virtual_mem_access_flags_t *>(data.data());
        ASSERT_TRUE(flags & UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE);
    } break;

    default:
        FAIL() << "Unhandled ur_virtual_mem_info_t enumeration: " << info;
        break;
    }
}

using urVirtualMemGetInfoTest = uur::urVirtualMemMappedTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urVirtualMemGetInfoTest);

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
    size_t info_size = 0;
    ASSERT_EQ_RESULT(urVirtualMemGetInfo(context, virtual_ptr, size,
                                         UR_VIRTUAL_MEM_INFO_FORCE_UINT32, 0,
                                         nullptr, &info_size),
                     UR_RESULT_ERROR_INVALID_ENUMERATION);
}
