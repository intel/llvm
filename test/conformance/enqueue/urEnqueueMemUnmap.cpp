// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

struct urEnqueueMemUnmapTest : public uur::urMemBufferQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urMemBufferQueueTest::SetUp());
        ASSERT_SUCCESS(urEnqueueMemBufferMap(
            queue, buffer, true, UR_MAP_FLAG_READ | UR_MAP_FLAG_WRITE, 0, size,
            0, nullptr, nullptr, (void **)&map));
    };

    void TearDown() override { uur::urMemBufferQueueTest::TearDown(); }

    uint32_t *map = nullptr;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueMemUnmapTest);

TEST_P(urEnqueueMemUnmapTest, Success) {
    ASSERT_SUCCESS(urEnqueueMemUnmap(queue, buffer, map, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemUnmapTest, InvalidNullHandleQueue) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urEnqueueMemUnmap(nullptr, buffer, map, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemUnmapTest, InvalidNullHandleMem) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urEnqueueMemUnmap(queue, nullptr, map, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemUnmapTest, InvalidNullPtrMap) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urEnqueueMemUnmap(queue, buffer, nullptr, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemUnmapTest, InvalidNullPtrEventWaitList) {
    ASSERT_EQ_RESULT(urEnqueueMemUnmap(queue, buffer, map, 1, nullptr, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(
        urEnqueueMemUnmap(queue, buffer, map, 0, &validEvent, nullptr),
        UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t inv_evt = nullptr;
    ASSERT_EQ_RESULT(
        urEnqueueMemUnmap(queue, buffer, map, 1, &inv_evt, nullptr),
        UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ASSERT_SUCCESS(urEventRelease(validEvent));
}
