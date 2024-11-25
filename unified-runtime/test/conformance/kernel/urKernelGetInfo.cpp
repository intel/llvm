// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ur_api.h"
#include <uur/fixtures.h>
#include <uur/known_failure.h>

using urKernelGetInfoTest = uur::urKernelTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urKernelGetInfoTest);

TEST_P(urKernelGetInfoTest, SuccessFunctionName) {
  const ur_kernel_info_t property_name = UR_KERNEL_INFO_FUNCTION_NAME;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urKernelGetInfo(kernel, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_GT(property_size, 0);

  std::vector<char> property_value(property_size, '\0');
  ASSERT_SUCCESS(urKernelGetInfo(kernel, property_name, property_size,
                                 property_value.data(), nullptr));

  ASSERT_TRUE(uur::stringPropertyIsValid(property_value.data(), property_size));
}

TEST_P(urKernelGetInfoTest, SuccessNumArgs) {
  const ur_kernel_info_t property_name = UR_KERNEL_INFO_NUM_ARGS;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urKernelGetInfo(kernel, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 999;
  ASSERT_SUCCESS(urKernelGetInfo(kernel, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_NE(property_value, 999);
}

TEST_P(urKernelGetInfoTest, SuccessReferenceCount) {
  const ur_kernel_info_t property_name = UR_KERNEL_INFO_REFERENCE_COUNT;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urKernelGetInfo(kernel, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 999;
  ASSERT_QUERY_RETURNS_VALUE(urKernelGetInfo(kernel, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);

  ASSERT_GT(property_value, 0U);
}

TEST_P(urKernelGetInfoTest, SuccessContext) {
  const ur_kernel_info_t property_name = UR_KERNEL_INFO_CONTEXT;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urKernelGetInfo(kernel, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_context_handle_t));

  ur_context_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urKernelGetInfo(kernel, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(context, property_value);
}

TEST_P(urKernelGetInfoTest, SuccessProgram) {
  const ur_kernel_info_t property_name = UR_KERNEL_INFO_PROGRAM;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urKernelGetInfo(kernel, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_program_handle_t));

  ur_program_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urKernelGetInfo(kernel, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(program, property_value);
}

TEST_P(urKernelGetInfoTest, SuccessAttributes) {
  const ur_kernel_info_t property_name = UR_KERNEL_INFO_ATTRIBUTES;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urKernelGetInfo(kernel, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_GT(property_size, 0);

  std::vector<char> property_value(property_size, '\0');
  ASSERT_SUCCESS(urKernelGetInfo(kernel, property_name, property_size,
                                 property_value.data(), nullptr));

  ASSERT_TRUE(uur::stringPropertyIsValid(property_value.data(), property_size));

  const std::string returned_attributes = std::string(property_value.data());
  ur_backend_t backend = UR_BACKEND_FORCE_UINT32;
  ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                   sizeof(backend), &backend, nullptr));
  if (backend == UR_BACKEND_OPENCL || backend == UR_BACKEND_LEVEL_ZERO) {
    // Older intel drivers don't attach any default attributes and newer
    // ones force walk order to X/Y/Z using special attribute.
    ASSERT_TRUE(returned_attributes.empty() ||
                returned_attributes ==
                    "intel_reqd_workgroup_walk_order(0,1,2)");
  } else {
    ASSERT_TRUE(returned_attributes.empty());
  }
}

TEST_P(urKernelGetInfoTest, SuccessNumRegs) {
  UUR_KNOWN_FAILURE_ON(uur::HIP{});

  const ur_kernel_info_t property_name = UR_KERNEL_INFO_NUM_REGS;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urKernelGetInfo(kernel, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 999;
  ASSERT_SUCCESS(urKernelGetInfo(kernel, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_NE(property_value, 999);
}

TEST_P(urKernelGetInfoTest, SuccessSpillMemSize) {
  UUR_KNOWN_FAILURE_ON(uur::HIP{}, uur::OpenCL{});

  ur_kernel_info_t property_name = UR_KERNEL_INFO_SPILL_MEM_SIZE;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urKernelGetInfo(kernel, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  std::vector<uint32_t> property_value(property_size / sizeof(uint32_t));
  ASSERT_SUCCESS(urKernelGetInfo(kernel, property_name, property_size,
                                 property_value.data(), nullptr));
}

TEST_P(urKernelGetInfoTest, InvalidNullHandleKernel) {
  size_t property_size = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urKernelGetInfo(nullptr, UR_KERNEL_INFO_FUNCTION_NAME, 0,
                                   nullptr, &property_size));
}

TEST_P(urKernelGetInfoTest, InvalidEnumeration) {
  size_t property_size = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urKernelGetInfo(kernel, UR_KERNEL_INFO_FORCE_UINT32, 0,
                                   nullptr, &property_size));
}

TEST_P(urKernelGetInfoTest, InvalidSizeZero) {
  size_t property_size = 0;
  ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS, 0, nullptr,
                                 &property_size));

  std::vector<char> property_value(property_size);
  ASSERT_EQ_RESULT(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS, 0,
                                   property_value.data(), nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urKernelGetInfoTest, InvalidSizeSmall) {
  size_t property_size = 0;
  ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS, 0, nullptr,
                                 &property_size));

  std::vector<char> property_value(property_size);
  ASSERT_EQ_RESULT(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS,
                                   property_value.size() - 1,
                                   property_value.data(), nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urKernelGetInfoTest, InvalidNullPointerPropValue) {
  size_t property_size = 0;
  ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS, 0, nullptr,
                                 &property_size));
  ASSERT_EQ_RESULT(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS,
                                   property_size, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urKernelGetInfoTest, InvalidNullPointerPropSizeRet) {
  ASSERT_EQ_RESULT(
      urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS, 0, nullptr, nullptr),
      UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urKernelGetInfoTest, KernelNameCorrect) {
  const ur_kernel_info_t property_name = UR_KERNEL_INFO_FUNCTION_NAME;
  size_t property_size = 0;

  ASSERT_SUCCESS(
      urKernelGetInfo(kernel, property_name, 0, nullptr, &property_size));
  ASSERT_GT(property_size, 0);

  std::vector<char> property_value(property_size, '\0');
  ASSERT_SUCCESS(urKernelGetInfo(kernel, property_name, property_size,
                                 property_value.data(), nullptr));

  ASSERT_TRUE(uur::stringPropertyIsValid(property_value.data(), property_size));
  ASSERT_STREQ(kernel_name.c_str(), property_value.data());
}

TEST_P(urKernelGetInfoTest, KernelContextCorrect) {
  ur_context_handle_t info_context;
  ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_CONTEXT,
                                 sizeof(ur_context_handle_t), &info_context,
                                 nullptr));
  ASSERT_EQ(context, info_context);
}
