// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ur_api.h"
#include "uur/checks.h"
#include "ze_api.h"
#include <uur/fixtures.h>

using urLevelZeroKernelNativeHandleTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urLevelZeroKernelNativeHandleTest);

TEST_P(urLevelZeroKernelNativeHandleTest, OwnedHandleRelease) {
  ze_context_handle_t native_context;
  urContextGetNativeHandle(context, (ur_native_handle_t *)&native_context);

  ze_device_handle_t native_device;
  urDeviceGetNativeHandle(device, (ur_native_handle_t *)&native_device);

  std::shared_ptr<std::vector<char>> il_binary;
  uur::KernelsEnvironment::instance->LoadSource("foo", platform, il_binary);

  auto kernel_name =
      uur::KernelsEnvironment::instance->GetEntryPointNames("foo")[0];

  ze_module_desc_t moduleDesc{};
  moduleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  moduleDesc.inputSize = il_binary->size();
  moduleDesc.pInputModule =
      reinterpret_cast<const uint8_t *>(il_binary->data());
  moduleDesc.pBuildFlags = "";
  ze_module_handle_t module;

  // Initialize Level Zero driver is required if this test is linked statically
  // with Level Zero loader, the driver will not be init otherwise.
  zeInit(ZE_INIT_FLAG_GPU_ONLY);

  ASSERT_EQ(
      zeModuleCreate(native_context, native_device, &moduleDesc, &module, NULL),
      ZE_RESULT_SUCCESS);

  ze_kernel_desc_t kernelDesc{};
  kernelDesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  kernelDesc.pKernelName = kernel_name.c_str();

  ze_kernel_handle_t native_kernel;

  ASSERT_EQ(zeKernelCreate(module, &kernelDesc, &native_kernel),
            ZE_RESULT_SUCCESS);

  ur_program_native_properties_t pprops = {
      UR_STRUCTURE_TYPE_PROGRAM_NATIVE_PROPERTIES, nullptr, 1};

  ur_program_handle_t program;
  ASSERT_SUCCESS(urProgramCreateWithNativeHandle((ur_native_handle_t)module,
                                                 context, &pprops, &program));

  ur_kernel_native_properties_t kprops = {
      UR_STRUCTURE_TYPE_KERNEL_NATIVE_PROPERTIES, nullptr, 1};

  ur_kernel_handle_t kernel;
  ASSERT_SUCCESS(urKernelCreateWithNativeHandle(
      (ur_native_handle_t)native_kernel, context, program, &kprops, &kernel));

  size_t global_offset = 0;
  size_t local_size = 1;
  size_t global_size = 1;
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, 1, &global_offset,
                                       &local_size, &global_size, 0, nullptr,
                                       nullptr));

  ASSERT_SUCCESS(urKernelRelease(kernel));
  ASSERT_SUCCESS(urProgramRelease(program));
}

TEST_P(urLevelZeroKernelNativeHandleTest, NullProgram) {
  ze_context_handle_t native_context;
  urContextGetNativeHandle(context, (ur_native_handle_t *)&native_context);

  ze_device_handle_t native_device;
  urDeviceGetNativeHandle(device, (ur_native_handle_t *)&native_device);

  std::shared_ptr<std::vector<char>> il_binary;
  uur::KernelsEnvironment::instance->LoadSource("foo", platform, il_binary);

  auto kernel_name =
      uur::KernelsEnvironment::instance->GetEntryPointNames("foo")[0];

  ze_module_desc_t moduleDesc{};
  moduleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  moduleDesc.inputSize = il_binary->size();
  moduleDesc.pInputModule =
      reinterpret_cast<const uint8_t *>(il_binary->data());
  moduleDesc.pBuildFlags = "";
  ze_module_handle_t module;

  ASSERT_EQ(
      zeModuleCreate(native_context, native_device, &moduleDesc, &module, NULL),
      ZE_RESULT_SUCCESS);

  ze_kernel_desc_t kernelDesc{};
  kernelDesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  kernelDesc.pKernelName = kernel_name.c_str();

  ze_kernel_handle_t native_kernel;

  ASSERT_EQ(zeKernelCreate(module, &kernelDesc, &native_kernel),
            ZE_RESULT_SUCCESS);

  ur_kernel_native_properties_t kprops = {
      UR_STRUCTURE_TYPE_KERNEL_NATIVE_PROPERTIES, nullptr, 1};

  ur_kernel_handle_t kernel;
  EXPECT_EQ(urKernelCreateWithNativeHandle((ur_native_handle_t)native_kernel,
                                           context, nullptr, &kprops, &kernel),
            UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}
