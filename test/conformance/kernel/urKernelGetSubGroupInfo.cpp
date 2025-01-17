// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include "ur_api.h"
#include "uur/known_failure.h"
#include <uur/fixtures.h>

struct urKernelGetSubGroupInfoFixedSubGroupSizeTest : uur::urKernelTest {
  void SetUp() override {
    UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{}, uur::OpenCL{},
                         uur::LevelZero{}, uur::LevelZeroV2{});
    program_name = "fixed_sg_size";
    UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::SetUp());
  }

  // This value correlates to sub_group_size<8> in fixed_sg_size.cpp.
  uint32_t num_sub_groups{8};
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(
    urKernelGetSubGroupInfoFixedSubGroupSizeTest);

TEST_P(urKernelGetSubGroupInfoFixedSubGroupSizeTest,
       SuccessCompileNumSubGroups) {
  ur_kernel_sub_group_info_t property_name =
      UR_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urKernelGetSubGroupInfo(kernel, device, property_name, 0, nullptr,
                              &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value;
  ASSERT_SUCCESS(urKernelGetSubGroupInfo(
      kernel, device, property_name, property_size, &property_value, nullptr));
  ASSERT_EQ(property_value, num_sub_groups);
}

struct urKernelGetSubGroupInfoTest : uur::urKernelTest {
  void SetUp() override { UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::SetUp()); }
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelGetSubGroupInfoTest);

TEST_P(urKernelGetSubGroupInfoTest, SuccessMaxSubGroupSize) {
  ur_kernel_sub_group_info_t property_name =
      UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urKernelGetSubGroupInfo(kernel, device, property_name, 0, nullptr,
                              &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  std::vector<char> property_value(property_size);
  ASSERT_SUCCESS(urKernelGetSubGroupInfo(kernel, device, property_name,
                                         property_size, property_value.data(),
                                         nullptr));
}

TEST_P(urKernelGetSubGroupInfoTest, SuccessMaxNumSubGroups) {
  ur_kernel_sub_group_info_t property_name =
      UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urKernelGetSubGroupInfo(kernel, device, property_name, 0, nullptr,
                              &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  std::vector<char> property_value(property_size);
  ASSERT_SUCCESS(urKernelGetSubGroupInfo(kernel, device, property_name,
                                         property_size, property_value.data(),
                                         nullptr));
}

TEST_P(urKernelGetSubGroupInfoTest, SuccessSubGroupSizeIntel) {
  ur_kernel_sub_group_info_t property_name =
      UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urKernelGetSubGroupInfo(kernel, device, property_name, 0, nullptr,
                              &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  std::vector<char> property_value(property_size);
  ASSERT_SUCCESS(urKernelGetSubGroupInfo(kernel, device, property_name,
                                         property_size, property_value.data(),
                                         nullptr));
}

TEST_P(urKernelGetSubGroupInfoTest, SuccessCompileNumSubgroupsIsZero) {
  // Returns 0 by default when there is no specific information
  size_t subgroups = 1;
  ASSERT_SUCCESS(urKernelGetSubGroupInfo(
      kernel, device, UR_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS,
      sizeof(subgroups), &subgroups, nullptr));
  ASSERT_EQ(subgroups, 0);
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
                       kernel, device, UR_KERNEL_SUB_GROUP_INFO_FORCE_UINT32, 0,
                       nullptr, &bad_enum_length));
}
