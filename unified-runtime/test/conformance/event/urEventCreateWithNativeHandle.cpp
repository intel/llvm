// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"
#include "uur/known_failure.h"
#include "uur/raii.h"

using urEventCreateWithNativeHandleTest = uur::event::urEventTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEventCreateWithNativeHandleTest);

TEST_P(urEventCreateWithNativeHandleTest, Success) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  ur_native_handle_t native_event = 0;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventGetNativeHandle(event, &native_event));

  // We cannot assume anything about a native_handle, not even if it's
  // `nullptr` since this could be a valid representation within a backend.
  // We can however convert the native_handle back into a unified-runtime handle
  // and perform some query on it to verify that it works.
  uur::raii::Event evt = nullptr;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventCreateWithNativeHandle(native_event, context, nullptr, evt.ptr()));
  ASSERT_NE(evt, nullptr);

  ur_execution_info_t exec_info;
  ASSERT_SUCCESS(urEventGetInfo(evt, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                                sizeof(ur_execution_info_t), &exec_info,
                                nullptr));
}

TEST_P(urEventCreateWithNativeHandleTest, SuccessWithProperties) {
  ur_native_handle_t native_event = 0;
  {
    UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
        urEventGetNativeHandle(event, &native_event));
  }

  uur::raii::Event evt = nullptr;
  // We can't pass isNativeHandleOwned = true in the generic tests since
  // we always get the native handle from a UR object, and transferring
  // ownership from one UR object to another isn't allowed.
  ur_event_native_properties_t props = {
      UR_STRUCTURE_TYPE_EVENT_NATIVE_PROPERTIES, nullptr, false};
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventCreateWithNativeHandle(native_event, context, &props, evt.ptr()));
  ASSERT_NE(evt, nullptr);

  ur_execution_info_t exec_info;
  ASSERT_SUCCESS(urEventGetInfo(evt, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                                sizeof(ur_execution_info_t), &exec_info,
                                nullptr));
}

TEST_P(urEventCreateWithNativeHandleTest, InvalidNullHandle) {
  ur_native_handle_t native_event = 0;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventGetNativeHandle(event, &native_event));

  uur::raii::Event evt = nullptr;
  ASSERT_EQ_RESULT(
      urEventCreateWithNativeHandle(native_event, nullptr, nullptr, evt.ptr()),
      UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEventCreateWithNativeHandleTest, InvalidNullPointer) {
  ur_native_handle_t native_event = 0;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urEventGetNativeHandle(event, &native_event));

  uur::raii::Event evt = nullptr;
  ASSERT_EQ_RESULT(
      urEventCreateWithNativeHandle(native_event, context, nullptr, nullptr),
      UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
