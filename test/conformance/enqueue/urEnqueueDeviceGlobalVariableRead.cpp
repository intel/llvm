// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception
#include <uur/fixtures.h>

using urEnqueueDeviceGetGlobalVariableReadTest = uur::urGlobalVariableTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueDeviceGetGlobalVariableReadTest);

TEST_P(urEnqueueDeviceGetGlobalVariableReadTest, Success) {

  ASSERT_SUCCESS(urEnqueueDeviceGlobalVariableWrite(
      queue, program, global_var.name.c_str(), true, sizeof(global_var.value),
      0, &global_var.value, 0, nullptr, nullptr));

  size_t global_offset = 0;
  size_t n_dimensions = 1;
  size_t global_size = 1;

  // execute the kernel
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                       &global_offset, &global_size, nullptr, 0,
                                       nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  // read global var back to host
  ASSERT_SUCCESS(urEnqueueDeviceGlobalVariableRead(
      queue, program, global_var.name.c_str(), true, sizeof(global_var.value),
      0, &global_var.value, 0, nullptr, nullptr));

  // kernel should increment value
  ASSERT_EQ(global_var.value, 1);
}

TEST_P(urEnqueueDeviceGetGlobalVariableReadTest, InvalidNullHandleQueue) {
  ASSERT_EQ_RESULT(urEnqueueDeviceGlobalVariableRead(
                       nullptr, program, global_var.name.c_str(), true,
                       sizeof(global_var.value), 0, &global_var.value, 0,
                       nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEnqueueDeviceGetGlobalVariableReadTest, InvalidNullHandleProgram) {
  ASSERT_EQ_RESULT(
      urEnqueueDeviceGlobalVariableRead(queue, nullptr, global_var.name.c_str(),
                                        true, sizeof(global_var.value), 0,
                                        &global_var.value, 0, nullptr, nullptr),
      UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEnqueueDeviceGetGlobalVariableReadTest, InvalidNullPointerName) {
  ASSERT_EQ_RESULT(urEnqueueDeviceGlobalVariableRead(
                       queue, program, nullptr, true, sizeof(global_var.value),
                       0, &global_var.value, 0, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urEnqueueDeviceGetGlobalVariableReadTest, InvalidNullPointerDst) {
  ASSERT_EQ_RESULT(
      urEnqueueDeviceGlobalVariableRead(queue, program, global_var.name.c_str(),
                                        true, sizeof(global_var.value), 0,
                                        nullptr, 0, nullptr, nullptr),
      UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urEnqueueDeviceGetGlobalVariableReadTest,
       InvalidEventWaitListNullEvents) {
  ASSERT_EQ_RESULT(
      urEnqueueDeviceGlobalVariableRead(queue, program, global_var.name.c_str(),
                                        true, sizeof(global_var.value), 0,
                                        &global_var.value, 1, nullptr, nullptr),
      UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
}

TEST_P(urEnqueueDeviceGetGlobalVariableReadTest, InvalidEventWaitListZeroSize) {
  ur_event_handle_t evt = nullptr;
  ASSERT_EQ_RESULT(
      urEnqueueDeviceGlobalVariableRead(queue, program, global_var.name.c_str(),
                                        true, sizeof(global_var.value), 0,
                                        &global_var.value, 0, &evt, nullptr),
      UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
}
TEST_P(urEnqueueDeviceGetGlobalVariableReadTest, InvalidEventWaitInvalidEvent) {
  ur_event_handle_t inv_evt = nullptr;
  ASSERT_EQ_RESULT(urEnqueueDeviceGlobalVariableRead(
                       queue, program, global_var.name.c_str(), true,
                       sizeof(global_var.value), 0, &global_var.value, 1,
                       &inv_evt, nullptr),
                   UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
}
