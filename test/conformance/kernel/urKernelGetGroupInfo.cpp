// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ur_api.h"
#include <array>
#include <uur/fixtures.h>

struct urKernelGetGroupInfoFixedWorkGroupSizeTest : uur::urKernelTest {
    void SetUp() override {
        program_name = "fixed_wg_size";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::SetUp());
    }

    // This value correlates to work_group_size<8, 4, 2> in fixed_wg_size.cpp.
    // In SYCL, the right-most dimension varies the fastest in linearization.
    // In UR, this is on the left, so we reverse the order of these values.
    std::array<size_t, 3> work_group_size{2, 4, 8};
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelGetGroupInfoFixedWorkGroupSizeTest);

struct urKernelGetGroupInfoMaxWorkGroupSizeTest : uur::urKernelTest {
    void SetUp() override {
        program_name = "max_wg_size";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::SetUp());
    }

    // These values correlate to max_work_group_size<8, 4, 2> and
    // max_linear_work_group_size<64> in max_wg_size.cpp.
    // In SYCL, the right-most dimension varies the fastest in linearization.
    // In UR, this is on the left, so we reverse the order of these values.
    std::array<size_t, 3> max_work_group_size{2, 4, 8};
    size_t max_linear_work_group_size{64};
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelGetGroupInfoMaxWorkGroupSizeTest);

using urKernelGetGroupInfoTest = uur::urKernelTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelGetGroupInfoTest);

TEST_P(urKernelGetGroupInfoTest, GlobalWorkSize) {
    auto property_name = UR_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE;
    size_t property_size = 0;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urKernelGetGroupInfo(kernel, device, property_name, 0, nullptr,
                             &property_size),
        property_name);
    ASSERT_EQ(property_size, 3 * sizeof(size_t));

    std::vector<char> property_value(property_size);
    ASSERT_SUCCESS(urKernelGetGroupInfo(kernel, device, property_name,
                                        property_size, property_value.data(),
                                        nullptr));
}

TEST_P(urKernelGetGroupInfoTest, WorkGroupSize) {
    auto property_name = UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE;
    size_t property_size = 0;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urKernelGetGroupInfo(kernel, device, property_name, 0, nullptr,
                             &property_size),
        property_name);
    ASSERT_EQ(property_size, sizeof(size_t));

    std::vector<char> property_value(property_size);
    ASSERT_SUCCESS(urKernelGetGroupInfo(kernel, device, property_name,
                                        property_size, property_value.data(),
                                        nullptr));
}

TEST_P(urKernelGetGroupInfoFixedWorkGroupSizeTest, CompileWorkGroupSize) {
    auto property_name = UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE;
    size_t property_size = 0;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urKernelGetGroupInfo(kernel, device, property_name, 0, nullptr,
                             &property_size),
        property_name);
    ASSERT_EQ(property_size, 3 * sizeof(size_t));

    std::array<size_t, 3> property_value;
    ASSERT_SUCCESS(urKernelGetGroupInfo(kernel, device, property_name,
                                        property_size, property_value.data(),
                                        nullptr));

    ASSERT_EQ(property_value, work_group_size);
}

TEST_P(urKernelGetGroupInfoTest, LocalMemSize) {
    auto property_name = UR_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE;
    size_t property_size = 0;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urKernelGetGroupInfo(kernel, device, property_name, 0, nullptr,
                             &property_size),
        property_name);
    ASSERT_EQ(property_size, sizeof(size_t));

    std::vector<char> property_value(property_size);
    ASSERT_SUCCESS(urKernelGetGroupInfo(kernel, device, property_name,
                                        property_size, property_value.data(),
                                        nullptr));
}

TEST_P(urKernelGetGroupInfoTest, PreferredWorkGroupSizeMultiple) {
    auto property_name =
        UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE;
    size_t property_size = 0;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urKernelGetGroupInfo(kernel, device, property_name, 0, nullptr,
                             &property_size),
        property_name);
    ASSERT_EQ(property_size, sizeof(size_t));

    std::vector<char> property_value(property_size);
    ASSERT_SUCCESS(urKernelGetGroupInfo(kernel, device, property_name,
                                        property_size, property_value.data(),
                                        nullptr));
}

TEST_P(urKernelGetGroupInfoTest, PrivateMemSize) {
    auto property_name = UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE;
    size_t property_size = 0;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urKernelGetGroupInfo(kernel, device, property_name, 0, nullptr,
                             &property_size),
        property_name);
    ASSERT_EQ(property_size, sizeof(size_t));

    std::vector<char> property_value(property_size);
    ASSERT_SUCCESS(urKernelGetGroupInfo(kernel, device, property_name,
                                        property_size, property_value.data(),
                                        nullptr));
}

TEST_P(urKernelGetGroupInfoMaxWorkGroupSizeTest, CompileMaxWorkGroupSize) {
    auto property_name = UR_KERNEL_GROUP_INFO_COMPILE_MAX_WORK_GROUP_SIZE;
    size_t property_size = 0;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urKernelGetGroupInfo(kernel, device, property_name, 0, nullptr,
                             &property_size),
        property_name);
    ASSERT_EQ(property_size, 3 * sizeof(size_t));

    std::array<size_t, 3> property_value;
    ASSERT_SUCCESS(urKernelGetGroupInfo(kernel, device, property_name,
                                        property_size, property_value.data(),
                                        nullptr));

    ASSERT_EQ(property_value, max_work_group_size);
}

TEST_P(urKernelGetGroupInfoMaxWorkGroupSizeTest,
       CompileMaxLinearWorkGroupSize) {
    auto property_name =
        UR_KERNEL_GROUP_INFO_COMPILE_MAX_LINEAR_WORK_GROUP_SIZE;
    size_t property_size = 0;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urKernelGetGroupInfo(kernel, device, property_name, 0, nullptr,
                             &property_size),
        property_name);
    ASSERT_EQ(property_size, sizeof(size_t));

    size_t property_value;
    ASSERT_SUCCESS(urKernelGetGroupInfo(kernel, device, property_name,
                                        property_size, &property_value,
                                        nullptr));

    ASSERT_EQ(property_value, max_linear_work_group_size);
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

TEST_P(urKernelGetGroupInfoTest, CompileWorkGroupSizeEmpty) {
    // Returns 0 by default when there is no specific information
    std::array<size_t, 3> read_dims{1, 1, 1};
    std::array<size_t, 3> zero{0, 0, 0};
    ASSERT_SUCCESS(urKernelGetGroupInfo(
        kernel, device, UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE,
        sizeof(read_dims), read_dims.data(), nullptr));
    ASSERT_EQ(read_dims, zero);
}

TEST_P(urKernelGetGroupInfoTest, CompileMaxWorkGroupSizeEmpty) {
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
