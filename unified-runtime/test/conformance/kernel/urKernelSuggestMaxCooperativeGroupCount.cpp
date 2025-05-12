// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urKernelSuggestMaxCooperativeGroupCountTest
    : uur::urKernelExecutionTest {
  void SetUp() override {
    program_name = "bar";

    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());

    ur_bool_t cooperative_kernel_support = false;
    ASSERT_SUCCESS(urDeviceGetInfo(device,
                                   UR_DEVICE_INFO_COOPERATIVE_KERNEL_SUPPORT,
                                   sizeof(cooperative_kernel_support),
                                   &cooperative_kernel_support, nullptr));
    if (!cooperative_kernel_support) {
      GTEST_SKIP() << "Cooperative launch is not supported.";
    }
  }

  uint32_t suggested_work_groups = 0;
  const uint32_t n_dimensions = 1;
  const size_t local_size = 1;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urKernelSuggestMaxCooperativeGroupCountTest);

TEST_P(urKernelSuggestMaxCooperativeGroupCountTest, Success) {
  ASSERT_SUCCESS(urKernelSuggestMaxCooperativeGroupCount(
      kernel, device, n_dimensions, &local_size, 0, &suggested_work_groups));
  ASSERT_GE(suggested_work_groups, 0);
}

TEST_P(urKernelSuggestMaxCooperativeGroupCountTest, InvalidNullHandleKernel) {
  ASSERT_EQ_RESULT(urKernelSuggestMaxCooperativeGroupCount(
                       nullptr, device, n_dimensions, &local_size, 0,
                       &suggested_work_groups),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urKernelSuggestMaxCooperativeGroupCountTest, InvalidNullHandleDevice) {
  ASSERT_EQ_RESULT(urKernelSuggestMaxCooperativeGroupCount(
                       kernel, nullptr, n_dimensions, &local_size, 0,
                       &suggested_work_groups),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urKernelSuggestMaxCooperativeGroupCountTest, InvalidWorkDimension) {
  // Only supports 1-3 dimensions.
  ASSERT_EQ_RESULT(
      urKernelSuggestMaxCooperativeGroupCount(kernel, device, 4, &local_size, 0,
                                              &suggested_work_groups),
      UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  ASSERT_EQ_RESULT(
      urKernelSuggestMaxCooperativeGroupCount(kernel, device, 0, &local_size, 0,
                                              &suggested_work_groups),
      UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  ASSERT_EQ_RESULT(
      urKernelSuggestMaxCooperativeGroupCount(
          kernel, device, UINT32_MAX, &local_size, 0, &suggested_work_groups),
      UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
}

TEST_P(urKernelSuggestMaxCooperativeGroupCountTest,
       InvalidNullPointerLocalSize) {
  ASSERT_EQ_RESULT(
      urKernelSuggestMaxCooperativeGroupCount(
          kernel, device, n_dimensions, nullptr, 0, &suggested_work_groups),
      UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urKernelSuggestMaxCooperativeGroupCountTest,
       InvalidNullPointerGroupCountRet) {
  ASSERT_EQ_RESULT(urKernelSuggestMaxCooperativeGroupCount(
                       kernel, device, n_dimensions, &local_size, 0, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
