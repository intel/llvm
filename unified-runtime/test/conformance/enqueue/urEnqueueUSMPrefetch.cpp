// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urEnqueueUSMPrefetchWithParamTest
    : uur::urUSMDeviceAllocTestWithParam<ur_usm_migration_flag_t> {
  void SetUp() override {
    // The setup for the parent fixture does a urQueueFlush, which isn't
    // supported by native cpu.
    UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});
    uur::urUSMDeviceAllocTestWithParam<ur_usm_migration_flag_t>::SetUp();
  }
};

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urEnqueueUSMPrefetchWithParamTest,
    ::testing::Values(UR_USM_MIGRATION_FLAG_DEFAULT),
    uur::deviceTestWithParamPrinter<ur_usm_migration_flag_t>);

TEST_P(urEnqueueUSMPrefetchWithParamTest, Success) {
  UUR_KNOWN_FAILURE_ON(
      // HIP and CUDA return UR_RESULT_ERROR_ADAPTER_SPECIFIC to issue a
      // warning about the hint being unsupported. The same applies for
      // subsequent fails in this file.
      // TODO: codify this in the spec and account for it in the CTS.
      uur::HIP{}, uur::CUDA{},
      // The setup for the parent fixture does a urQueueFlush, which isn't
      // supported by native cpu. Again same goes for subsequent fails in
      // this file.
      uur::NativeCPU{});

  ur_event_handle_t prefetch_event = nullptr;
  ASSERT_SUCCESS(urEnqueueUSMPrefetch(queue, ptr, allocation_size, getParam(),
                                      0, nullptr, &prefetch_event));

  ASSERT_SUCCESS(urQueueFlush(queue));
  ASSERT_SUCCESS(urEventWait(1, &prefetch_event));

  ur_event_status_t event_status;
  ASSERT_SUCCESS(uur::GetEventInfo<ur_event_status_t>(
      prefetch_event, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS, event_status));
  ASSERT_EQ(event_status, UR_EVENT_STATUS_COMPLETE);
  ASSERT_SUCCESS(urEventRelease(prefetch_event));
}

/**
 * Tests that urEnqueueUSMPrefetch() waits for its dependencies to finish before
 * executing.
 */
TEST_P(urEnqueueUSMPrefetchWithParamTest, CheckWaitEvent) {
  UUR_KNOWN_FAILURE_ON(uur::HIP{}, uur::CUDA{}, uur::NativeCPU{});

  ur_queue_handle_t fill_queue;
  ASSERT_SUCCESS(urQueueCreate(context, device, nullptr, &fill_queue));

  size_t big_allocation = 65536;
  uint8_t fill_pattern = 0;
  void *fill_ptr = nullptr;
  ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                  big_allocation, &fill_ptr));

  ur_event_handle_t fill_event;
  ASSERT_SUCCESS(urEnqueueUSMFill(fill_queue, fill_ptr, 1, &fill_pattern,
                                  big_allocation, 0, nullptr, &fill_event));

  ur_event_handle_t prefetch_event = nullptr;
  ASSERT_SUCCESS(urEnqueueUSMPrefetch(queue, ptr, allocation_size, getParam(),
                                      1, &fill_event, &prefetch_event));

  ASSERT_SUCCESS(urQueueFlush(queue));
  ASSERT_SUCCESS(urQueueFlush(fill_queue));
  ASSERT_SUCCESS(urEventWait(1, &prefetch_event));

  ur_event_status_t memset_status;
  ASSERT_SUCCESS(uur::GetEventInfo<ur_event_status_t>(
      fill_event, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS, memset_status));
  ASSERT_EQ(memset_status, UR_EVENT_STATUS_COMPLETE);

  ur_event_status_t event_status;
  ASSERT_SUCCESS(uur::GetEventInfo<ur_event_status_t>(
      prefetch_event, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS, event_status));
  ASSERT_EQ(event_status, UR_EVENT_STATUS_COMPLETE);

  ASSERT_SUCCESS(urEventRelease(prefetch_event));
  ASSERT_SUCCESS(urEventRelease(fill_event));
  ASSERT_SUCCESS(urQueueRelease(fill_queue));

  ASSERT_SUCCESS(urUSMFree(context, fill_ptr));
}

struct urEnqueueUSMPrefetchTest : uur::urUSMDeviceAllocTest {
  void SetUp() override {
    // The setup for the parent fixture does a urQueueFlush, which isn't
    // supported by native cpu.
    UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});
    UUR_RETURN_ON_FATAL_FAILURE(uur::urUSMDeviceAllocTest::SetUp());
  }
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urEnqueueUSMPrefetchTest);

TEST_P(urEnqueueUSMPrefetchTest, InvalidNullHandleQueue) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urEnqueueUSMPrefetch(nullptr, ptr, allocation_size,
                                        UR_USM_MIGRATION_FLAG_DEFAULT, 0,
                                        nullptr, nullptr));
}

TEST_P(urEnqueueUSMPrefetchTest, InvalidNullPointerMem) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urEnqueueUSMPrefetch(queue, nullptr, allocation_size,
                                        UR_USM_MIGRATION_FLAG_DEFAULT, 0,
                                        nullptr, nullptr));
}

TEST_P(urEnqueueUSMPrefetchTest, InvalidEnumeration) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urEnqueueUSMPrefetch(queue, ptr, allocation_size,
                                        UR_USM_MIGRATION_FLAG_FORCE_UINT32, 0,
                                        nullptr, nullptr));
}

TEST_P(urEnqueueUSMPrefetchTest, InvalidSizeZero) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urEnqueueUSMPrefetch(queue, ptr, 0,
                                        UR_USM_MIGRATION_FLAG_DEFAULT, 0,
                                        nullptr, nullptr));
}

TEST_P(urEnqueueUSMPrefetchTest, InvalidSizeTooLarge) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{}, uur::NativeCPU{});

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urEnqueueUSMPrefetch(queue, ptr, allocation_size * 2,
                                        UR_USM_MIGRATION_FLAG_DEFAULT, 0,
                                        nullptr, nullptr));
}

TEST_P(urEnqueueUSMPrefetchTest, InvalidEventWaitList) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                   urEnqueueUSMPrefetch(queue, ptr, allocation_size,
                                        UR_USM_MIGRATION_FLAG_DEFAULT, 1,
                                        nullptr, nullptr));

  ur_event_handle_t validEvent;
  ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                   urEnqueueUSMPrefetch(queue, ptr, allocation_size,
                                        UR_USM_MIGRATION_FLAG_DEFAULT, 0,
                                        &validEvent, nullptr));

  ur_event_handle_t inv_evt = nullptr;
  ASSERT_EQ_RESULT(urEnqueueUSMPrefetch(queue, ptr, allocation_size,
                                        UR_USM_MIGRATION_FLAG_DEFAULT, 1,
                                        &inv_evt, nullptr),
                   UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

  ASSERT_SUCCESS(urEventRelease(validEvent));
}
