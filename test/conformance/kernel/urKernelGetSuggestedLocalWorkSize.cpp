// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urKernelGetSuggestedLocalWorkSizeTest : uur::urKernelExecutionTest {
  void SetUp() override {
    program_name = "bar";
    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
  }
  size_t global_size = 32;
  size_t global_offset = 0;
  size_t n_dimensions = 1;

  size_t suggested_local_work_size;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelGetSuggestedLocalWorkSizeTest);

TEST_P(urKernelGetSuggestedLocalWorkSizeTest, Success) {
  suggested_local_work_size = SIZE_MAX;
  auto result = urKernelGetSuggestedLocalWorkSize(kernel, queue, n_dimensions,
                                                  &global_offset, &global_size,
                                                  &suggested_local_work_size);
  if (result == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    GTEST_SKIP();
  }
  ASSERT_SUCCESS(result);
  ASSERT_LE(suggested_local_work_size, global_size);
}

TEST_P(urKernelGetSuggestedLocalWorkSizeTest, Success2D) {
  size_t global_size_2d[2] = {32, 32};
  size_t global_offset_2d[2] = {0, 0};
  size_t suggested_local_work_size_2d[2] = {SIZE_MAX, SIZE_MAX};
  auto result = urKernelGetSuggestedLocalWorkSize(
      kernel, queue, 2, global_offset_2d, global_size_2d,
      suggested_local_work_size_2d);
  if (result == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    GTEST_SKIP();
  }
  ASSERT_SUCCESS(result);
  for (int I = 0; I < 2; ++I) {
    ASSERT_LE(suggested_local_work_size_2d[I], global_size_2d[I]);
  }
}

TEST_P(urKernelGetSuggestedLocalWorkSizeTest, Success3D) {
  size_t global_size_3d[3] = {32, 32, 32};
  size_t global_offset_3d[3] = {0, 0, 0};
  size_t suggested_local_work_size_3d[3] = {SIZE_MAX, SIZE_MAX, SIZE_MAX};
  auto result = urKernelGetSuggestedLocalWorkSize(
      kernel, queue, 3, global_offset_3d, global_size_3d,
      suggested_local_work_size_3d);
  if (result == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    GTEST_SKIP();
  }
  ASSERT_SUCCESS(result);
  for (int I = 0; I < 3; ++I) {
    ASSERT_LE(suggested_local_work_size_3d[I], global_size_3d[I]);
  }
}

TEST_P(urKernelGetSuggestedLocalWorkSizeTest, InvalidNullHandleKernel) {
  ASSERT_EQ_RESULT(urKernelGetSuggestedLocalWorkSize(
                       nullptr, queue, n_dimensions, &global_offset,
                       &global_size, &suggested_local_work_size),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urKernelGetSuggestedLocalWorkSizeTest, InvalidNullHandleQueue) {
  ASSERT_EQ_RESULT(urKernelGetSuggestedLocalWorkSize(
                       kernel, nullptr, n_dimensions, &global_offset,
                       &global_size, &suggested_local_work_size),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urKernelGetSuggestedLocalWorkSizeTest, InvalidWorkDimension) {
  uint32_t max_work_item_dimensions = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(
      device, UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS,
      sizeof(max_work_item_dimensions), &max_work_item_dimensions, nullptr));
  auto result = urKernelGetSuggestedLocalWorkSize(
      kernel, queue, max_work_item_dimensions + 1, &global_offset, &global_size,
      &suggested_local_work_size);
  if (result == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    GTEST_SKIP();
  }
  ASSERT_EQ_RESULT(result, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
}

TEST_P(urKernelGetSuggestedLocalWorkSizeTest, InvalidGlobalOffset) {
  ASSERT_EQ_RESULT(urKernelGetSuggestedLocalWorkSize(
                       kernel, queue, n_dimensions, nullptr, &global_size,
                       &suggested_local_work_size),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urKernelGetSuggestedLocalWorkSizeTest, InvalidGlobalSize) {
  ASSERT_EQ_RESULT(
      urKernelGetSuggestedLocalWorkSize(kernel, queue, n_dimensions,
                                        &global_offset, nullptr, nullptr),
      UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urKernelGetSuggestedLocalWorkSizeTest, InvalidSuggestedLocalWorkSize) {
  ASSERT_EQ_RESULT(
      urKernelGetSuggestedLocalWorkSize(kernel, queue, n_dimensions,
                                        &global_offset, &global_size, nullptr),
      UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
