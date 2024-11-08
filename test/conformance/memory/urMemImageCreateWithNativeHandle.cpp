// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urMemImageCreateWithNativeHandleTest : uur::urMemImageTest {
  void SetUp() {
    UUR_KNOWN_FAILURE_ON(uur::LevelZeroV2{});
    UUR_RETURN_ON_FATAL_FAILURE(urMemImageTest::SetUp());
  }
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemImageCreateWithNativeHandleTest);

TEST_P(urMemImageCreateWithNativeHandleTest, Success) {
  UUR_KNOWN_FAILURE_ON(uur::HIP{}, uur::LevelZero{});
  ur_native_handle_t native_handle = 0;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urMemGetNativeHandle(image, device, &native_handle));

  ur_mem_handle_t mem = nullptr;
  ASSERT_SUCCESS(urMemImageCreateWithNativeHandle(
      native_handle, context, &image_format, &image_desc, nullptr, &mem));
  ASSERT_NE(nullptr, mem);

  ur_context_handle_t mem_context = nullptr;
  ASSERT_SUCCESS(urMemGetInfo(mem, UR_MEM_INFO_CONTEXT,
                              sizeof(ur_context_handle_t), &mem_context,
                              nullptr));
  ASSERT_EQ(context, mem_context);
}

TEST_P(urMemImageCreateWithNativeHandleTest, SuccessWithProperties) {
  ur_native_handle_t native_handle = 0;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urMemGetNativeHandle(image, device, &native_handle));

  ur_mem_handle_t mem = nullptr;
  ur_mem_native_properties_t props = {UR_STRUCTURE_TYPE_MEM_NATIVE_PROPERTIES,
                                      nullptr, false};
  // We can't pass isNativeHandleOwned = true in the generic tests since
  // we always get the native handle from a UR object, and transferring
  // ownership from one UR object to another isn't allowed.
  ASSERT_SUCCESS(urMemImageCreateWithNativeHandle(
      native_handle, context, &image_format, &image_desc, &props, &mem));
  ASSERT_NE(nullptr, mem);

  ur_context_handle_t mem_context = nullptr;
  ASSERT_SUCCESS(urMemGetInfo(mem, UR_MEM_INFO_CONTEXT,
                              sizeof(ur_context_handle_t), &mem_context,
                              nullptr));
  ASSERT_EQ(context, mem_context);
}

TEST_P(urMemImageCreateWithNativeHandleTest, InvalidNullHandle) {
  ur_native_handle_t native_handle = 0;
  ASSERT_SUCCESS(urMemGetNativeHandle(image, device, &native_handle));

  ur_mem_handle_t mem = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urMemImageCreateWithNativeHandle(native_handle, nullptr,
                                                    &image_format, &image_desc,
                                                    nullptr, &mem));
}

TEST_P(urMemImageCreateWithNativeHandleTest, InvalidNullPointer) {
  ur_native_handle_t native_handle = 0;
  ASSERT_SUCCESS(urMemGetNativeHandle(image, device, &native_handle));

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urMemImageCreateWithNativeHandle(native_handle, context,
                                                    &image_format, &image_desc,
                                                    nullptr, nullptr));
}
