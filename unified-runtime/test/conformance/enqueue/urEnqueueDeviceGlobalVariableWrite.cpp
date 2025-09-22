// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urEnqueueDeviceGetGlobalVariableWriteWithParamTest =
    uur::urGlobalVariableWithParamTest<uur::BoolTestParam>;

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urEnqueueDeviceGetGlobalVariableWriteWithParamTest,
    testing::ValuesIn(uur::BoolTestParam::makeBoolParam("Blocking")),
    uur::deviceTestWithParamPrinter<uur::BoolTestParam>);

TEST_P(urEnqueueDeviceGetGlobalVariableWriteWithParamTest, Success) {
  bool is_blocking = getParam().value;
  global_var.value = 42;

  ASSERT_SUCCESS(urEnqueueDeviceGlobalVariableWrite(
      queue, program, global_var.name.c_str(), is_blocking,
      sizeof(global_var.value), 0, &global_var.value, 0, nullptr, nullptr));

  if (!is_blocking) {
    ASSERT_SUCCESS(urQueueFinish(queue));
  }

  size_t global_offset = 0;
  size_t n_dimensions = 1;
  size_t global_size = 1;

  // execute the kernel
  ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                       &global_offset, &global_size, nullptr, 0,
                                       nullptr, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  // read global var back to host
  int return_value = 0;
  ASSERT_SUCCESS(urEnqueueDeviceGlobalVariableRead(
      queue, program, global_var.name.c_str(), true, sizeof(return_value), 0,
      &return_value, 0, nullptr, nullptr));

  // kernel should return global_var.value + 1
  ASSERT_EQ(return_value, global_var.value + 1);
}

using urEnqueueDeviceGetGlobalVariableWriteTest = uur::urGlobalVariableTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEnqueueDeviceGetGlobalVariableWriteTest);

TEST_P(urEnqueueDeviceGetGlobalVariableWriteTest, InvalidNullHandleQueue) {
  ASSERT_EQ_RESULT(urEnqueueDeviceGlobalVariableWrite(
                       nullptr, program, global_var.name.c_str(), true,
                       sizeof(global_var.value), 0, &global_var.value, 0,
                       nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEnqueueDeviceGetGlobalVariableWriteTest, InvalidNullHandleProgram) {
  ASSERT_EQ_RESULT(urEnqueueDeviceGlobalVariableWrite(
                       queue, nullptr, global_var.name.c_str(), true,
                       sizeof(global_var.value), 0, &global_var.value, 0,
                       nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEnqueueDeviceGetGlobalVariableWriteTest, InvalidNullPointerName) {
  ASSERT_EQ_RESULT(urEnqueueDeviceGlobalVariableWrite(
                       queue, program, nullptr, true, sizeof(global_var.value),
                       0, &global_var.value, 0, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urEnqueueDeviceGetGlobalVariableWriteTest, InvalidNullPointerSrc) {
  ASSERT_EQ_RESULT(urEnqueueDeviceGlobalVariableWrite(
                       queue, program, global_var.name.c_str(), true,
                       sizeof(global_var.value), 0, nullptr, 0, nullptr,
                       nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urEnqueueDeviceGetGlobalVariableWriteTest,
       InvalidEventWaitListNullEvents) {
  ASSERT_EQ_RESULT(urEnqueueDeviceGlobalVariableWrite(
                       queue, program, global_var.name.c_str(), true,
                       sizeof(global_var.value), 0, &global_var.value, 1,
                       nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
}

TEST_P(urEnqueueDeviceGetGlobalVariableWriteTest,
       InvalidEventWaitListZeroSize) {
  ur_event_handle_t evt = nullptr;
  ASSERT_EQ_RESULT(urEnqueueDeviceGlobalVariableWrite(
                       queue, program, global_var.name.c_str(), true,
                       sizeof(global_var.value), 0, &global_var.value, 0, &evt,
                       nullptr),
                   UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
}
TEST_P(urEnqueueDeviceGetGlobalVariableWriteTest,
       InvalidEventWaitInvalidEvent) {
  ur_event_handle_t inv_evt = nullptr;
  ASSERT_EQ_RESULT(urEnqueueDeviceGlobalVariableWrite(
                       queue, program, global_var.name.c_str(), true,
                       sizeof(global_var.value), 0, &global_var.value, 1,
                       &inv_evt, nullptr),
                   UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
}
