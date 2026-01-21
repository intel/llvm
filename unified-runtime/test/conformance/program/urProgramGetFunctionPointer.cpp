// Copyright (C) 2023-2026 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/known_failure.h"
#include <uur/fixtures.h>

struct urProgramGetFunctionPointerTest : uur::urProgramTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
    ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
    auto kernel_names = uur::KernelsEnvironment::instance->GetEntryPointNames(
        this->program_name);
    function_name = kernel_names[0];
  }

  std::string function_name;
};

UUR_DEVICE_TEST_SUITE_WITH_DEFAULT_QUEUE(urProgramGetFunctionPointerTest);

TEST_P(urProgramGetFunctionPointerTest, Success) {
  void *function_pointer = nullptr;
  ur_result_t res = urProgramGetFunctionPointer(
      device, program, function_name.data(), &function_pointer);
  if (res == UR_RESULT_ERROR_FUNCTION_ADDRESS_NOT_AVAILABLE ||
      res == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    return;
  }
  ASSERT_SUCCESS(res);
  ASSERT_NE(function_pointer, nullptr);
}

TEST_P(urProgramGetFunctionPointerTest, InvalidKernelName) {
  void *function_pointer = nullptr;
  std::string missing_function = "aFakeFunctionName";
  auto result = urProgramGetFunctionPointer(
      device, program, missing_function.data(), &function_pointer);
  if (result == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    return;
  }
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_KERNEL_NAME, result);
  ASSERT_EQ(function_pointer, nullptr);
}

TEST_P(urProgramGetFunctionPointerTest, InvalidNullHandleDevice) {
  void *function_pointer = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urProgramGetFunctionPointer(nullptr, program,
                                               function_name.data(),
                                               &function_pointer));
}

TEST_P(urProgramGetFunctionPointerTest, InvalidNullHandleProgram) {
  void *function_pointer = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urProgramGetFunctionPointer(device, nullptr,
                                               function_name.data(),
                                               &function_pointer));
}

TEST_P(urProgramGetFunctionPointerTest, InvalidNullPointerFunctionName) {
  void *function_pointer = nullptr;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_POINTER,
      urProgramGetFunctionPointer(device, program, nullptr, &function_pointer));
}

TEST_P(urProgramGetFunctionPointerTest, InvalidNullPointerFunctionPointer) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urProgramGetFunctionPointer(device, program,
                                               function_name.data(), nullptr));
}
