// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %use-mock %validate leaks-test | FileCheck %s

#include "fixtures.hpp"

#include <ur_mock_helpers.hpp>
#include <uur/fixtures.h>

// We need a fake handle for the below adapter leak test.
inline ur_result_t fakeAdapter_urAdapterGet(void *pParams) {
  const auto &params = *static_cast<ur_adapter_get_params_t *>(pParams);
  **params.pphAdapters = reinterpret_cast<ur_adapter_handle_t>(0x1);
  return UR_RESULT_SUCCESS;
}

struct adapterLeakTest : public urTest {
  void SetUp() override {
    urTest::SetUp();
    mock::getCallbacks().set_replace_callback("urAdapterGet",
                                              &fakeAdapter_urAdapterGet);
    mock::getCallbacks().set_replace_callback("urAdapterRetain",
                                              &genericSuccessCallback);
  }

  void TearDown() override {
    mock::getCallbacks().resetCallbacks();
    urTest::TearDown();
  }
};

inline ur_result_t fakeQueue_urQueueCreate(void *pParams) {
  const auto &params = *static_cast<ur_queue_create_params_t *>(pParams);
  **params.pphQueue = reinterpret_cast<ur_queue_handle_t>(0x4a);
  return UR_RESULT_SUCCESS;
}

inline ur_result_t fakeQueue_urEnqueueEventsWait(void *pParams) {
  const auto &params = *static_cast<ur_enqueue_events_wait_params_t *>(pParams);
  **params.pphEvent = reinterpret_cast<ur_event_handle_t>(0x4b);
  return UR_RESULT_SUCCESS;
}

struct queueLeakTest : public valDeviceTest {
  void SetUp() override {
    valDeviceTest::SetUp();
    mock::getCallbacks().set_replace_callback("urQueueCreate",
                                              &fakeQueue_urQueueCreate);
    mock::getCallbacks().set_replace_callback("urQueueRelease",
                                              &genericSuccessCallback);
    mock::getCallbacks().set_replace_callback("urEnqueueEventsWait",
                                              &fakeQueue_urEnqueueEventsWait);
    mock::getCallbacks().set_replace_callback("urEventRelease",
                                              &genericSuccessCallback);

    ASSERT_EQ(urContextCreate(1, &device, nullptr, &context),
              UR_RESULT_SUCCESS);
    ASSERT_NE(nullptr, context);

    ASSERT_EQ(urQueueCreate(context, device, nullptr, &queue),
              UR_RESULT_SUCCESS);
    ASSERT_NE(nullptr, context);
  }

  void TearDown() override {
    ASSERT_EQ(urQueueRelease(queue), UR_RESULT_SUCCESS);
    ASSERT_EQ(urContextRelease(context), UR_RESULT_SUCCESS);

    mock::getCallbacks().resetCallbacks();
    valDeviceTest::TearDown();
  }

  ur_context_handle_t context;
  ur_queue_handle_t queue;
};

// CHECK: [ RUN      ] adapterLeakTest.testUrAdapterGetLeak
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[ERROR]: Retained 1 reference(s) to handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[ERROR]: Handle {{[0-9xa-fA-F]+}} was recorded for first time here:
TEST_F(adapterLeakTest, testUrAdapterGetLeak) {
  ur_adapter_handle_t adapter = nullptr;
  ASSERT_EQ(urAdapterGet(1, &adapter, nullptr), UR_RESULT_SUCCESS);
  ASSERT_NE(nullptr, adapter);
}

// CHECK: [ RUN      ] adapterLeakTest.testUrAdapterRetainLeak
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 2
// CHECK-NEXT: <VALIDATION>[ERROR]: Retained 2 reference(s) to handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[ERROR]: Handle {{[0-9xa-fA-F]+}} was recorded for first time here:
TEST_F(adapterLeakTest, testUrAdapterRetainLeak) {
  ur_adapter_handle_t adapter = nullptr;
  ASSERT_EQ(urAdapterGet(1, &adapter, nullptr), UR_RESULT_SUCCESS);
  ASSERT_NE(nullptr, adapter);
  ASSERT_EQ(urAdapterRetain(adapter), UR_RESULT_SUCCESS);
}

// CHECK: [ RUN      ] adapterLeakTest.testUrAdapterRetainNonexistent
// CHECK-NEXT: <VALIDATION>[ERROR]: Attempting to retain nonexistent handle {{[0-9xa-fA-F]+}}
TEST_F(adapterLeakTest, testUrAdapterRetainNonexistent) {
  ur_adapter_handle_t adapter = (ur_adapter_handle_t)0xBEEF;
  ASSERT_EQ(urAdapterRetain(adapter), UR_RESULT_SUCCESS);
  ASSERT_NE(nullptr, adapter);
}

// CHECK: [ RUN      ] valDeviceTest.testUrContextCreateLeak
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[ERROR]: Retained 1 reference(s) to handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[ERROR]: Handle {{[0-9xa-fA-F]+}} was recorded for first time here:
TEST_F(valDeviceTest, testUrContextCreateLeak) {
  ur_context_handle_t context = nullptr;
  ASSERT_EQ(urContextCreate(1, &device, nullptr, &context), UR_RESULT_SUCCESS);
  ASSERT_NE(nullptr, context);
}

// CHECK: [ RUN      ] valDeviceTest.testUrContextRetainLeak
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 2
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[ERROR]: Retained 2 reference(s) to handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[ERROR]: Handle {{[0-9xa-fA-F]+}} was recorded for first time here:
TEST_F(valDeviceTest, testUrContextRetainLeak) {
  ur_context_handle_t context = nullptr;
  ASSERT_EQ(urContextCreate(1, &device, nullptr, &context), UR_RESULT_SUCCESS);
  ASSERT_NE(nullptr, context);
  ASSERT_EQ(urContextRetain(context), UR_RESULT_SUCCESS);
}

// CHECK: [ RUN      ] valDeviceTest.testUrContextRetainNonexistent
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[ERROR]: Attempting to retain nonexistent handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
TEST_F(valDeviceTest, testUrContextRetainNonexistent) {
  ur_context_handle_t context = (ur_context_handle_t)0xC0FFEE;
  ASSERT_EQ(urContextRetain(context), UR_RESULT_SUCCESS);
}

// CHECK: [ RUN      ] valDeviceTest.testUrContextCreateSuccess
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
TEST_F(valDeviceTest, testUrContextCreateSuccess) {
  ur_context_handle_t context = nullptr;
  ASSERT_EQ(urContextCreate(1, &device, nullptr, &context), UR_RESULT_SUCCESS);
  ASSERT_NE(nullptr, context);
  ASSERT_EQ(urContextRelease(context), UR_RESULT_SUCCESS);
}

// CHECK: [ RUN      ] valDeviceTest.testUrContextRetainSuccess
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 2
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
TEST_F(valDeviceTest, testUrContextRetainSuccess) {
  ur_context_handle_t context = nullptr;
  ASSERT_EQ(urContextCreate(1, &device, nullptr, &context), UR_RESULT_SUCCESS);
  ASSERT_NE(nullptr, context);
  ASSERT_EQ(urContextRetain(context), UR_RESULT_SUCCESS);
  ASSERT_EQ(urContextRelease(context), UR_RESULT_SUCCESS);
  ASSERT_EQ(urContextRelease(context), UR_RESULT_SUCCESS);
}

// CHECK: [ RUN      ] valDeviceTest.testUrContextReleaseLeak
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[ERROR]: Attempting to release nonexistent handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to -1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[ERROR]: Retained -1 reference(s) to handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[ERROR]: Handle {{[0-9xa-fA-F]+}} was recorded for first time here:
TEST_F(valDeviceTest, testUrContextReleaseLeak) {
  ur_context_handle_t context = nullptr;
  ASSERT_EQ(urContextCreate(1, &device, nullptr, &context), UR_RESULT_SUCCESS);
  ASSERT_NE(nullptr, context);
  ASSERT_EQ(urContextRelease(context), UR_RESULT_SUCCESS);
  ASSERT_EQ(urContextRelease(context), UR_RESULT_SUCCESS);
}

// CHECK: [ RUN      ] valDeviceTest.testUrContextReleaseNonexistent
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[ERROR]: Attempting to release nonexistent handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to -1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[ERROR]: Retained -1 reference(s) to handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[ERROR]: Handle {{[0-9xa-fA-F]+}} was recorded for first time here:
TEST_F(valDeviceTest, testUrContextReleaseNonexistent) {
  ur_context_handle_t context = (ur_context_handle_t)0xC0FFEE;
  ASSERT_EQ(urContextRelease(context), UR_RESULT_SUCCESS);
}

// CHECK: [ RUN      ] queueLeakTest.testUrEnqueueSuccess
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0

TEST_F(queueLeakTest, testUrEnqueueSuccess) {
  ur_event_handle_t event = nullptr;
  ASSERT_EQ(urEnqueueEventsWait(queue, 0, nullptr, &event), UR_RESULT_SUCCESS);
  ASSERT_EQ(urEventRelease(event), UR_RESULT_SUCCESS);
}

// CHECK: [ RUN      ] queueLeakTest.testUrEnqueueLeak
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[ERROR]: Retained 1 reference(s) to handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[ERROR]: Handle {{[0-9xa-fA-F]+}} was recorded for first time here:
TEST_F(queueLeakTest, testUrEnqueueLeak) {
  ur_event_handle_t event = nullptr;
  ASSERT_EQ(urEnqueueEventsWait(queue, 0, nullptr, &event), UR_RESULT_SUCCESS);
}

// CHECK: [ RUN      ] queueLeakTest.testUrEventReleaseNonexistent
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 1
// CHECK-NEXT: <VALIDATION>[ERROR]: Attempting to release nonexistent handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to -1
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[DEBUG]: Reference count for handle {{[0-9xa-fA-F]+}} changed to 0
// CHECK-NEXT: <VALIDATION>[ERROR]: Retained -1 reference(s) to handle {{[0-9xa-fA-F]+}}
// CHECK-NEXT: <VALIDATION>[ERROR]: Handle {{[0-9xa-fA-F]+}} was recorded for first time here:
TEST_F(queueLeakTest, testUrEventReleaseNonexistent) {
  ur_event_handle_t event = (ur_event_handle_t)0xBEEF;
  ASSERT_EQ(urEventRelease(event), UR_RESULT_SUCCESS);
}
