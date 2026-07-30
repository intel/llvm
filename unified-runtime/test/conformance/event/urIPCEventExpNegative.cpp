// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ipc_event_fixtures.h"

using urIPCEventExpNegativeTest = uur::event::urIPCEventTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urIPCEventExpNegativeTest);

TEST_P(urIPCEventExpNegativeTest, GetWithNullOutData) {
  size_t size = 0;
  EXPECT_EQ_RESULT(urIPCGetEventHandleExp(event, nullptr, &size),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urIPCEventExpNegativeTest, GetWithNullOutSize) {
  void *data = nullptr;
  EXPECT_EQ_RESULT(urIPCGetEventHandleExp(event, &data, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urIPCEventExpNegativeTest, GetWithNullEvent) {
  void *data = nullptr;
  size_t size = 0;
  EXPECT_EQ_RESULT(urIPCGetEventHandleExp(nullptr, &data, &size),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urIPCEventExpNegativeTest, OpenWithMismatchedSize) {
  void *data = nullptr;
  size_t size = 0;
  ASSERT_SUCCESS(urIPCGetEventHandleExp(event, &data, &size));

  ur_event_handle_t imported = nullptr;
  EXPECT_EQ_RESULT(urIPCOpenEventHandleExp(context, data, size + 1, &imported),
                   UR_RESULT_ERROR_INVALID_VALUE);
  EXPECT_EQ_RESULT(urIPCOpenEventHandleExp(context, data, size - 1, &imported),
                   UR_RESULT_ERROR_INVALID_VALUE);

  EXPECT_SUCCESS(urIPCPutEventHandleExp(context, data));
}

TEST_P(urIPCEventExpNegativeTest, OpenWithNullData) {
  ur_event_handle_t imported = nullptr;
  EXPECT_EQ_RESULT(urIPCOpenEventHandleExp(context, nullptr, 64, &imported),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urIPCEventExpNegativeTest, OpenWithNullPhEvent) {
  void *data = nullptr;
  size_t size = 0;
  ASSERT_SUCCESS(urIPCGetEventHandleExp(event, &data, &size));

  EXPECT_EQ_RESULT(urIPCOpenEventHandleExp(context, data, size, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
  EXPECT_SUCCESS(urIPCPutEventHandleExp(context, data));
}

TEST_P(urIPCEventExpNegativeTest, OpenWithNullContext) {
  void *data = nullptr;
  size_t size = 0;
  ASSERT_SUCCESS(urIPCGetEventHandleExp(event, &data, &size));

  ur_event_handle_t imported = nullptr;
  EXPECT_EQ_RESULT(urIPCOpenEventHandleExp(nullptr, data, size, &imported),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  EXPECT_SUCCESS(urIPCPutEventHandleExp(context, data));
}

TEST_P(urIPCEventExpNegativeTest, PutWithNullData) {
  EXPECT_EQ_RESULT(urIPCPutEventHandleExp(context, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

// Profiling + IPC at create time is mutually exclusive.
TEST_P(urIPCEventExpNegativeTest, CreateRejectsProfilingPlusIPC) {
  ur_exp_event_desc_t desc{UR_STRUCTURE_TYPE_EXP_EVENT_DESC, nullptr,
                           UR_EXP_EVENT_FLAG_IPC_EXP |
                               UR_EXP_EVENT_FLAG_ENABLE_PROFILING};

  ur_event_handle_t evt = nullptr;
  EXPECT_EQ_RESULT(urEventCreateExp(context, device, &desc, &evt),
                   UR_RESULT_ERROR_INVALID_VALUE);
  if (evt) {
    EXPECT_SUCCESS(urEventRelease(evt));
  }
}

// Reserved bits outside the documented flags must be rejected.
TEST_P(urIPCEventExpNegativeTest, CreateRejectsReservedFlagBits) {
  ur_exp_event_desc_t desc{UR_STRUCTURE_TYPE_EXP_EVENT_DESC, nullptr,
                           UR_EXP_EVENT_FLAG_IPC_EXP |
                               static_cast<ur_exp_event_flags_t>(0x80000000)};

  ur_event_handle_t evt = nullptr;
  EXPECT_EQ_RESULT(urEventCreateExp(context, device, &desc, &evt),
                   UR_RESULT_ERROR_INVALID_ENUMERATION);
  if (evt) {
    EXPECT_SUCCESS(urEventRelease(evt));
  }
}

// Get rejects re-export of an already-imported event.
TEST_P(urIPCEventExpNegativeTest, GetRejectsImportedEvent) {
  void *data = nullptr;
  size_t size = 0;
  ASSERT_SUCCESS(urIPCGetEventHandleExp(event, &data, &size));

  ur_event_handle_t imported = nullptr;
  ASSERT_SUCCESS(urIPCOpenEventHandleExp(context, data, size, &imported));

  void *reexport = nullptr;
  size_t reexportSize = 0;
  EXPECT_EQ_RESULT(urIPCGetEventHandleExp(imported, &reexport, &reexportSize),
                   UR_RESULT_ERROR_INVALID_EVENT);

  EXPECT_SUCCESS(urEventRelease(imported));
  EXPECT_SUCCESS(urIPCPutEventHandleExp(context, data));
}
