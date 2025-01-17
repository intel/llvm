// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urProgramSetSpecializationConstantsTest : uur::urKernelExecutionTest {
  void SetUp() override {
    program_name = "spec_constant";
    UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
  }

  uint32_t spec_value = 42;
  uint32_t default_spec_value = 1000; // Must match the one in the SYCL source
  ur_specialization_constant_info_t info = {0, sizeof(spec_value), &spec_value};
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urProgramSetSpecializationConstantsTest);

struct urProgramSetMultipleSpecializationConstantsTest
    : uur::urKernelExecutionTest {
  // The types of spec constants in this program are {uint32_t, uint64_t, bool}
  void SetUp() override {
    program_name = "spec_constant_multiple";
    UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
  }
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(
    urProgramSetMultipleSpecializationConstantsTest);

TEST_P(urProgramSetSpecializationConstantsTest, Success) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{});

  ASSERT_SUCCESS(urProgramSetSpecializationConstants(program, 1, &info));
  ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
  auto entry_points =
      uur::KernelsEnvironment::instance->GetEntryPointNames(program_name);
  kernel_name = entry_points[0];
  ASSERT_SUCCESS(urKernelCreate(program, kernel_name.data(), &kernel));

  ur_mem_handle_t buffer;
  AddBuffer1DArg(sizeof(spec_value), &buffer);
  Launch1DRange(1);
  ValidateBuffer<uint32_t>(buffer, sizeof(spec_value), spec_value);
}

TEST_P(urProgramSetSpecializationConstantsTest, UseDefaultValue) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{});

  ur_platform_backend_t backend;
  ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                   sizeof(ur_platform_backend_t), &backend,
                                   nullptr));
  if (backend == UR_PLATFORM_BACKEND_CUDA ||
      backend == UR_PLATFORM_BACKEND_HIP) {
    GTEST_FAIL() << "This test is known to cause crashes on Nvidia and "
                    "AMD; not running.";
  }

  ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
  auto entry_points =
      uur::KernelsEnvironment::instance->GetEntryPointNames(program_name);
  kernel_name = entry_points[0];
  ASSERT_SUCCESS(urKernelCreate(program, kernel_name.data(), &kernel));

  ur_mem_handle_t buffer;
  AddBuffer1DArg(sizeof(spec_value), &buffer);
  Launch1DRange(1);
  ValidateBuffer<uint32_t>(buffer, sizeof(spec_value), default_spec_value);
}

TEST_P(urProgramSetMultipleSpecializationConstantsTest, MultipleCalls) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{});

  uint32_t a = 100;
  uint64_t b = 200;
  bool c = false;
  uint64_t output = 0;

  ur_specialization_constant_info_t info_a = {0, sizeof(uint32_t), &a};
  ASSERT_SUCCESS(urProgramSetSpecializationConstants(program, 1, &info_a));

  ur_specialization_constant_info_t info_c = {2, sizeof(bool), &c};
  ASSERT_SUCCESS(urProgramSetSpecializationConstants(program, 1, &info_c));

  ur_specialization_constant_info_t info_b = {1, sizeof(uint64_t), &b};
  ASSERT_SUCCESS(urProgramSetSpecializationConstants(program, 1, &info_b));

  ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
  auto entry_points =
      uur::KernelsEnvironment::instance->GetEntryPointNames(program_name);
  kernel_name = entry_points[0];
  ASSERT_SUCCESS(urKernelCreate(program, kernel_name.data(), &kernel));

  ur_mem_handle_t buffer;
  AddBuffer1DArg(sizeof(uint64_t), &buffer);
  Launch1DRange(1);

  ASSERT_SUCCESS(urEnqueueMemBufferRead(
      queue, buffer, true, 0, sizeof(uint64_t), &output, 0, nullptr, nullptr));
  ASSERT_EQ(output, 300);
}

TEST_P(urProgramSetMultipleSpecializationConstantsTest, SingleCall) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{});

  uint32_t a = 200;
  uint64_t b = 300;
  bool c = true;
  uint64_t output = 0;

  ur_specialization_constant_info_t info[3] = {
      {0, sizeof(uint32_t), &a},
      {2, sizeof(bool), &c},
      {1, sizeof(uint64_t), &b},
  };
  ASSERT_SUCCESS(urProgramSetSpecializationConstants(program, 3, &info[0]));

  ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
  auto entry_points =
      uur::KernelsEnvironment::instance->GetEntryPointNames(program_name);
  kernel_name = entry_points[0];
  ASSERT_SUCCESS(urKernelCreate(program, kernel_name.data(), &kernel));

  ur_mem_handle_t buffer;
  AddBuffer1DArg(sizeof(uint64_t), &buffer);
  Launch1DRange(1);

  ASSERT_SUCCESS(urEnqueueMemBufferRead(
      queue, buffer, true, 0, sizeof(uint64_t), &output, 0, nullptr, nullptr));
  ASSERT_EQ(output, 100);
}

TEST_P(urProgramSetSpecializationConstantsTest, InvalidNullHandleProgram) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urProgramSetSpecializationConstants(nullptr, 1, &info));
}

TEST_P(urProgramSetSpecializationConstantsTest,
       InvalidNullPointerSpecConstants) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urProgramSetSpecializationConstants(program, 1, nullptr));
}

TEST_P(urProgramSetSpecializationConstantsTest, InvalidSizeCount) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urProgramSetSpecializationConstants(program, 0, &info));
}

TEST_P(urProgramSetSpecializationConstantsTest, InvalidValueSize) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{}, uur::LevelZero{},
                       uur::LevelZeroV2{});

  ur_specialization_constant_info_t bad_info = {0, 0x1000, &spec_value};
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_VALUE,
                   urProgramSetSpecializationConstants(program, 1, &bad_info));
}

TEST_P(urProgramSetSpecializationConstantsTest, InvalidValueId) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{}, uur::LevelZero{},
                       uur::LevelZeroV2{});

  ur_specialization_constant_info_t bad_info = {999, sizeof(spec_value),
                                                &spec_value};
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SPEC_ID,
                   urProgramSetSpecializationConstants(program, 1, &bad_info));
}

TEST_P(urProgramSetSpecializationConstantsTest, InvalidValuePtr) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{}, uur::LevelZero{},
                       uur::LevelZeroV2{});

  ur_specialization_constant_info_t bad_info = {0, sizeof(spec_value), nullptr};
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_VALUE,
                   urProgramSetSpecializationConstants(program, 1, &bad_info));
}
