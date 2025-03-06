// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urKernelCreateWithNativeHandleTest : uur::urKernelTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::SetUp());

    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
        urKernelGetNativeHandle(kernel, &native_kernel_handle));
  }

  ur_native_handle_t native_kernel_handle = 0;
  ur_kernel_handle_t native_kernel = nullptr;
  // We can't pass isNativeHandleOwned = true in the generic tests since
  // we always get the native handle from a UR object, and transferring
  // ownership from one UR object to another isn't allowed.
  ur_kernel_native_properties_t properties = {
      UR_STRUCTURE_TYPE_KERNEL_NATIVE_PROPERTIES, /*sType*/
      nullptr,                                    /*pNext*/
      false                                       /*isNativeHandleOwned*/
  };
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urKernelCreateWithNativeHandleTest);

TEST_P(urKernelCreateWithNativeHandleTest, Success) {
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urKernelCreateWithNativeHandle(
      native_kernel_handle, context, program, nullptr, &native_kernel));

  uint32_t ref_count = 0;
  ASSERT_SUCCESS(urKernelGetInfo(native_kernel, UR_KERNEL_INFO_REFERENCE_COUNT,
                                 sizeof(ref_count), &ref_count, nullptr));

  ASSERT_NE(ref_count, 0);
}

TEST_P(urKernelCreateWithNativeHandleTest, SuccessWithProperties) {
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urKernelCreateWithNativeHandle(
      native_kernel_handle, context, program, &properties, &native_kernel));

  uint32_t ref_count = 0;
  ASSERT_SUCCESS(urKernelGetInfo(native_kernel, UR_KERNEL_INFO_REFERENCE_COUNT,
                                 sizeof(ref_count), &ref_count, nullptr));

  ASSERT_NE(ref_count, 0);
}

TEST_P(urKernelCreateWithNativeHandleTest, InvalidNullHandleContext) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urKernelCreateWithNativeHandle(native_kernel_handle, nullptr,
                                                  program, &properties,
                                                  &native_kernel));
}

TEST_P(urKernelCreateWithNativeHandleTest, InvalidNullPointerNativeKernel) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urKernelCreateWithNativeHandle(native_kernel_handle, context,
                                                  program, &properties,
                                                  nullptr));
}
