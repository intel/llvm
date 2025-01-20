// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urKernelSetSpecializationConstantsTest : uur::urBaseKernelExecutionTest {
  void SetUp() override {
    program_name = "spec_constant";
    UUR_RETURN_ON_FATAL_FAILURE(urBaseKernelExecutionTest::SetUp());
    bool supports_kernel_spec_constant = false;
    ASSERT_SUCCESS(urDeviceGetInfo(
        device, UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS,
        sizeof(supports_kernel_spec_constant), &supports_kernel_spec_constant,
        nullptr));
    if (!supports_kernel_spec_constant) {
      GTEST_SKIP() << "Device does not support setting kernel spec constants.";
    }
    Build();
  }

  uint32_t spec_value = 42;
  ur_specialization_constant_info_t info = {0, sizeof(spec_value), &spec_value};
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelSetSpecializationConstantsTest);

struct urKernelSetSpecializationConstantsNegativeTest
    : uur::urBaseKernelExecutionTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urBaseKernelExecutionTest::SetUp());
    bool supports_kernel_spec_constant = false;
    ASSERT_SUCCESS(urDeviceGetInfo(
        device, UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS,
        sizeof(supports_kernel_spec_constant), &supports_kernel_spec_constant,
        nullptr));
    if (supports_kernel_spec_constant) {
      GTEST_SKIP() << "Device supports setting kernel spec constants.";
    }
    Build();
  }

  uint32_t spec_value = 42;
  ur_specialization_constant_info_t info = {0, sizeof(spec_value), &spec_value};
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(
    urKernelSetSpecializationConstantsNegativeTest);

TEST_P(urKernelSetSpecializationConstantsTest, Success) {
  ASSERT_SUCCESS(urKernelSetSpecializationConstants(kernel, 1, &info));

  ur_mem_handle_t buffer;
  AddBuffer1DArg(sizeof(spec_value), &buffer);
  Launch1DRange(1);
  ValidateBuffer<uint32_t>(buffer, sizeof(spec_value), spec_value);
}

TEST_P(urKernelSetSpecializationConstantsNegativeTest, Unsupported) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_UNSUPPORTED_FEATURE,
                   urKernelSetSpecializationConstants(kernel, 1, &info));
}

TEST_P(urKernelSetSpecializationConstantsTest, InvalidNullHandleKernel) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urKernelSetSpecializationConstants(nullptr, 1, &info));
}

TEST_P(urKernelSetSpecializationConstantsTest,
       InvalidNullPointerSpecConstants) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urKernelSetSpecializationConstants(kernel, 1, nullptr));
}

TEST_P(urKernelSetSpecializationConstantsTest, InvalidSizeCount) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urKernelSetSpecializationConstants(kernel, 0, &info));
}

TEST_P(urKernelSetSpecializationConstantsTest, InvalidValueSize) {
  ur_specialization_constant_info_t bad_info = {0, 0x1000, &spec_value};
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_VALUE,
                   urKernelSetSpecializationConstants(kernel, 1, &bad_info));
}

TEST_P(urKernelSetSpecializationConstantsTest, InvalidValueId) {
  ur_specialization_constant_info_t bad_info = {999, sizeof(spec_value),
                                                &spec_value};
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SPEC_ID,
                   urKernelSetSpecializationConstants(kernel, 1, &bad_info));
}

TEST_P(urKernelSetSpecializationConstantsTest, InvalidValuePtr) {
  ur_specialization_constant_info_t bad_info = {0, sizeof(spec_value), nullptr};
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_VALUE,
                   urKernelSetSpecializationConstants(kernel, 1, &bad_info));
}
