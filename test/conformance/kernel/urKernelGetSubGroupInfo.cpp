// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

using urKernelGetSubGroupInfoTest =
    uur::urKernelTestWithParam<ur_kernel_sub_group_info_t>;

UUR_TEST_SUITE_P(
    urKernelGetSubGroupInfoTest,
    ::testing::Values(UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE,
                      UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS,
                      UR_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS,
                      UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL),
    uur::deviceTestWithParamPrinter<ur_kernel_sub_group_info_t>);

struct urKernelGetSubGroupInfoSingleTest : uur::urKernelTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::SetUp());
    }
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelGetSubGroupInfoSingleTest);

TEST_P(urKernelGetSubGroupInfoTest, Success) {
    auto property_name = getParam();
    size_t property_size = 0;
    std::vector<char> property_value;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urKernelGetSubGroupInfo(kernel, device, property_name, 0, nullptr,
                                &property_size),
        property_name);
    property_value.resize(property_size);
    ASSERT_SUCCESS(urKernelGetSubGroupInfo(kernel, device, property_name,
                                           property_size, property_value.data(),
                                           nullptr));
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
                         kernel, device, UR_KERNEL_SUB_GROUP_INFO_FORCE_UINT32,
                         0, nullptr, &bad_enum_length));
}

TEST_P(urKernelGetSubGroupInfoSingleTest, CompileNumSubgroupsIsZero) {
    // Returns 0 by default when there is no specific information
    size_t subgroups = 1;
    ASSERT_SUCCESS(urKernelGetSubGroupInfo(
        kernel, device, UR_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS,
        sizeof(subgroups), &subgroups, nullptr));
    ASSERT_EQ(subgroups, 0);
}
