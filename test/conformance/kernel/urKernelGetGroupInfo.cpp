// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
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
                      UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE,
                      UR_KERNEL_GROUP_INFO_COMPILE_MAX_WORK_GROUP_SIZE,
                      UR_KERNEL_GROUP_INFO_COMPILE_MAX_LINEAR_WORK_GROUP_SIZE),
    uur::deviceTestWithParamPrinter<ur_kernel_group_info_t>);

struct urKernelGetGroupInfoSingleTest : uur::urKernelTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::SetUp());
    }
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelGetGroupInfoSingleTest);

struct urKernelGetGroupInfoWgSizeTest : uur::urKernelTest {
    void SetUp() override {
        program_name = "fixed_wg_size";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::SetUp());
    }

    // This must match the size in fixed_wg_size.cpp
    std::array<size_t, 3> wg_size{4, 4, 4};
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelGetGroupInfoWgSizeTest);

TEST_P(urKernelGetGroupInfoTest, Success) {
    auto property_name = getParam();
    size_t property_size = 0;
    std::vector<char> property_value;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urKernelGetGroupInfo(kernel, device, property_name, 0, nullptr,
                             &property_size),
        property_name);
    property_value.resize(property_size);
    ASSERT_SUCCESS(urKernelGetGroupInfo(kernel, device, property_name,
                                        property_size, property_value.data(),
                                        nullptr));
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

TEST_P(urKernelGetGroupInfoWgSizeTest, CompileWorkGroupSize) {
    std::array<size_t, 3> read_dims{1, 1, 1};
    ASSERT_SUCCESS(urKernelGetGroupInfo(
        kernel, device, UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE,
        sizeof(read_dims), read_dims.data(), nullptr));
    ASSERT_EQ(read_dims, wg_size);
}

TEST_P(urKernelGetGroupInfoSingleTest, CompileWorkGroupSizeEmpty) {
    // Returns 0 by default when there is no specific information
    std::array<size_t, 3> read_dims{1, 1, 1};
    std::array<size_t, 3> zero{0, 0, 0};
    ASSERT_SUCCESS(urKernelGetGroupInfo(
        kernel, device, UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE,
        sizeof(read_dims), read_dims.data(), nullptr));
    ASSERT_EQ(read_dims, zero);
}

TEST_P(urKernelGetGroupInfoSingleTest, CompileMaxWorkGroupSizeEmpty) {
    // Returns 0 by default when there is no specific information
    std::array<size_t, 3> read_dims{1, 1, 1};
    std::array<size_t, 3> zero{0, 0, 0};
    auto result = urKernelGetGroupInfo(
        kernel, device, UR_KERNEL_GROUP_INFO_COMPILE_MAX_WORK_GROUP_SIZE,
        sizeof(read_dims), read_dims.data(), nullptr);
    if (result == UR_RESULT_SUCCESS) {
        ASSERT_EQ(read_dims, zero);
    } else {
        ASSERT_EQ(result, UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION);
    }
}
