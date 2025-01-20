// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urEnqueueDeviceGetGlobalVariableWriteTest = uur::urGlobalVariableTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueDeviceGetGlobalVariableWriteTest);

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
