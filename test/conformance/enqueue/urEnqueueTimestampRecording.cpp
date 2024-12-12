// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urEnqueueTimestampRecordingExpTest : uur::urQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());
        bool timestamp_recording_support = false;
        ASSERT_SUCCESS(uur::GetTimestampRecordingSupport(
            device, timestamp_recording_support));
        if (!timestamp_recording_support) {
            GTEST_SKIP() << "Timestamp recording is not supported";
        }
    }

    void TearDown() override { urQueueTest::TearDown(); }
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueTimestampRecordingExpTest);

void common_check(ur_event_handle_t event) {
    // All successful runs should return a non-zero profiling results.
    uint64_t queuedTime = 0, submitTime = 0, startTime = 0, endTime = 0;
    ASSERT_SUCCESS(
        urEventGetProfilingInfo(event, UR_PROFILING_INFO_COMMAND_QUEUED,
                                sizeof(uint64_t), &queuedTime, nullptr));
    ASSERT_SUCCESS(
        urEventGetProfilingInfo(event, UR_PROFILING_INFO_COMMAND_SUBMIT,
                                sizeof(uint64_t), &submitTime, nullptr));
    ASSERT_SUCCESS(
        urEventGetProfilingInfo(event, UR_PROFILING_INFO_COMMAND_START,
                                sizeof(uint64_t), &startTime, nullptr));
    ASSERT_SUCCESS(urEventGetProfilingInfo(event, UR_PROFILING_INFO_COMMAND_END,
                                           sizeof(uint64_t), &endTime,
                                           nullptr));
    ASSERT_GT(queuedTime, 0);
    ASSERT_GT(submitTime, 0);
    ASSERT_GT(startTime, 0);
    ASSERT_GT(endTime, 0);
    ASSERT_EQ(queuedTime, submitTime);
    ASSERT_EQ(startTime, endTime);
    ASSERT_GE(endTime, submitTime);
}

TEST_P(urEnqueueTimestampRecordingExpTest, Success) {
    UUR_KNOWN_FAILURE_ON(uur::HIP{}, uur::CUDA{});

    ur_event_handle_t event = nullptr;
    ASSERT_SUCCESS(
        urEnqueueTimestampRecordingExp(queue, false, 0, nullptr, &event));
    ASSERT_SUCCESS(urQueueFinish(queue));
    common_check(event);
    ASSERT_SUCCESS(urEventRelease(event));
}

TEST_P(urEnqueueTimestampRecordingExpTest, SuccessBlocking) {
    UUR_KNOWN_FAILURE_ON(uur::LevelZeroV2{}, uur::HIP{}, uur::CUDA{});

    ur_event_handle_t event = nullptr;
    ASSERT_SUCCESS(
        urEnqueueTimestampRecordingExp(queue, true, 0, nullptr, &event));
    common_check(event);
    ASSERT_SUCCESS(urEventRelease(event));
}

TEST_P(urEnqueueTimestampRecordingExpTest, InvalidNullHandleQueue) {
    ur_event_handle_t event = nullptr;
    ASSERT_EQ_RESULT(
        urEnqueueTimestampRecordingExp(nullptr, false, 0, nullptr, &event),
        UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEnqueueTimestampRecordingExpTest, InvalidNullPointerEvent) {
    ASSERT_EQ_RESULT(
        urEnqueueTimestampRecordingExp(queue, false, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urEnqueueTimestampRecordingExpTest, InvalidNullPtrEventWaitList) {
    ur_event_handle_t event = nullptr;
    ASSERT_EQ_RESULT(
        urEnqueueTimestampRecordingExp(queue, true, 1, nullptr, &event),
        UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));
    ASSERT_EQ_RESULT(
        urEnqueueTimestampRecordingExp(queue, true, 0, &validEvent, &event),
        UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
    ASSERT_SUCCESS(urEventRelease(validEvent));

    ur_event_handle_t invalidEvent = nullptr;
    ASSERT_EQ_RESULT(
        urEnqueueTimestampRecordingExp(queue, true, 0, &invalidEvent, &event),
        UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
}
