// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urKernelGetGroupInfoTest =
    uur::urKernelTestWithParam<ur_kernel_group_info_t>;

UUR_TEST_SUITE_P(
    urKernelGetGroupInfoTest,
    ::testing::Values(UR_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE,
                      UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE,
                      UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE,
                      UR_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE,
                      UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                      UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE),
    uur::deviceTestWithParamPrinter<ur_kernel_group_info_t>);

TEST_P(urKernelGetGroupInfoTest, Success) {
    auto property_name = getParam();
    size_t property_size = 0;
    std::vector<char> property_value;
    auto result = urKernelGetGroupInfo(kernel, device, property_name, 0,
                                       nullptr, &property_size);
    if (result == UR_RESULT_SUCCESS) {
        property_value.resize(property_size);
        ASSERT_SUCCESS(urKernelGetGroupInfo(kernel, device, property_name,
                                            property_size,
                                            property_value.data(), nullptr));
    } else {
        ASSERT_EQ_RESULT(result, UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION);
    }
}

TEST_P(urKernelGetGroupInfoTest, InvalidNullHandleKernel) {
    size_t work_group_size = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelGetGroupInfo(
                         nullptr, device, UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE,
                         sizeof(work_group_size), &work_group_size, nullptr));
}

TEST_P(urKernelGetGroupInfoTest, InvalidNullHandleDevice) {
    size_t work_group_size = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelGetGroupInfo(
                         kernel, nullptr, UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE,
                         sizeof(work_group_size), &work_group_size, nullptr));
}

TEST_P(urKernelGetGroupInfoTest, InvalidEnumeration) {
    size_t bad_enum_length = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urKernelGetGroupInfo(kernel, device,
                                          UR_KERNEL_GROUP_INFO_FORCE_UINT32, 0,
                                          nullptr, &bad_enum_length));
}
