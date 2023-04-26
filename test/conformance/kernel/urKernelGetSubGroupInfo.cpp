// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urKernelGetSubGroupInfoTest = uur::urKernelTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelGetSubGroupInfoTest);

TEST_P(urKernelGetSubGroupInfoTest, Success) {
    uint32_t max_num_sub_groups = 0;
    ASSERT_SUCCESS(urKernelGetSubGroupInfo(
        kernel, device, UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS,
        sizeof(max_num_sub_groups), &max_num_sub_groups, nullptr));

    uint32_t device_max_num_sub_groups = 0;
    ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS,
                                   sizeof(device_max_num_sub_groups),
                                   &device_max_num_sub_groups, nullptr));

    ASSERT_TRUE(max_num_sub_groups <= device_max_num_sub_groups);
}

TEST_P(urKernelGetSubGroupInfoTest, InvalidNullHandleKernel) {
    uint32_t max_num_sub_groups = 0;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urKernelGetSubGroupInfo(
            nullptr, device, UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS,
            sizeof(max_num_sub_groups), &max_num_sub_groups, nullptr));
}

TEST_P(urKernelGetSubGroupInfoTest, InvalidNullHandleDevice) {
    uint32_t max_num_sub_groups = 0;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urKernelGetSubGroupInfo(
            kernel, nullptr, UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS,
            sizeof(max_num_sub_groups), &max_num_sub_groups, nullptr));
}

TEST_P(urKernelGetSubGroupInfoTest, InvalidEnumeration) {
    size_t bad_enum_length = 0;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                     urKernelGetSubGroupInfo(
                         nullptr, device, UR_KERNEL_SUB_GROUP_INFO_FORCE_UINT32,
                         0, nullptr, &bad_enum_length));
}
