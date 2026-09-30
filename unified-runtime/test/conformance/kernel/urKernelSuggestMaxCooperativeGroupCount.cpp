// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

#include <array>

struct urKernelSuggestMaxCooperativeGroupCountTest
    : uur::urKernelExecutionTest {
  void SetUp() override {
    program_name = "bar";

    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());

    ur_kernel_launch_properties_flags_t supported_properties = 0;
    ASSERT_SUCCESS(urDeviceGetInfo(
        device, UR_DEVICE_INFO_KERNEL_LAUNCH_CAPABILITIES,
        sizeof(supported_properties), &supported_properties, nullptr));
    if (!(supported_properties &
          UR_KERNEL_LAUNCH_PROPERTIES_FLAG_COOPERATIVE)) {
      GTEST_SKIP() << "Cooperative launch is not supported.";
    }
  }

  uint32_t suggested_work_groups = 0;
  const uint32_t n_dimensions = 1;
  const size_t local_size = 1;
};

UUR_DEVICE_TEST_SUITE_WITH_DEFAULT_QUEUE(
    urKernelSuggestMaxCooperativeGroupCountTest);

TEST_P(urKernelSuggestMaxCooperativeGroupCountTest, Success) {
  ASSERT_SUCCESS(urKernelSuggestMaxCooperativeGroupCount(
      kernel, device, n_dimensions, &local_size, 0, &suggested_work_groups));
  ASSERT_GE(suggested_work_groups, 0);
}

TEST_P(urKernelSuggestMaxCooperativeGroupCountTest, DynamicSharedMemory) {
  uint64_t local_memory_size = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_LOCAL_MEM_SIZE,
                                 sizeof(local_memory_size), &local_memory_size,
                                 nullptr));
  uint32_t compute_unit_count = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_NUM_COMPUTE_UNITS,
                                 sizeof(compute_unit_count),
                                 &compute_unit_count, nullptr));

  ASSERT_SUCCESS(urKernelSuggestMaxCooperativeGroupCount(
      kernel, device, n_dimensions, &local_size, 0, &suggested_work_groups));

  // Check that increasing the dynamic shared memory size reduces
  // the number of suggested work groups.
  constexpr std::array<uint32_t, 4> divs = {8, 4, 2, 1};
  uint32_t previous_suggested_work_groups = suggested_work_groups;
  for (const uint32_t i : divs) {
    const size_t dynamic_shared_memory_size = local_memory_size / i;
    ASSERT_SUCCESS(urKernelSuggestMaxCooperativeGroupCount(
        kernel, device, n_dimensions, &local_size, dynamic_shared_memory_size,
        &suggested_work_groups));

    ASSERT_LE(suggested_work_groups, previous_suggested_work_groups);
    ASSERT_LE((uint64_t)suggested_work_groups,
              (uint64_t)(compute_unit_count)*i);
    previous_suggested_work_groups = suggested_work_groups;
  }
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
