// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urEnqueueReadHostPipeTest = uur::urHostPipeTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueReadHostPipeTest);

TEST_P(urEnqueueReadHostPipeTest, InvalidNullHandleQueue) {
    bool blocking = true;
    uint32_t numEventsInWaitList = 0;
    ur_event_handle_t phEventWaitList;
    ur_event_handle_t *phEvent = nullptr;

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueReadHostPipe(
                         nullptr, program, pipe_symbol, blocking, &buffer, size,
                         numEventsInWaitList, &phEventWaitList, phEvent));
}

TEST_P(urEnqueueReadHostPipeTest, InvalidNullHandleProgram) {
    bool blocking = true;
    uint32_t numEventsInWaitList = 0;
    ur_event_handle_t phEventWaitList;
    ur_event_handle_t *phEvent = nullptr;

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueReadHostPipe(
                         queue, nullptr, pipe_symbol, blocking, &buffer, size,
                         numEventsInWaitList, &phEventWaitList, phEvent));
}

TEST_P(urEnqueueReadHostPipeTest, InvalidNullPointerPipeSymbol) {
    bool blocking = true;
    uint32_t numEventsInWaitList = 0;
    ur_event_handle_t phEventWaitList;
    ur_event_handle_t *phEvent = nullptr;

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueReadHostPipe(queue, program, nullptr, blocking,
                                           &buffer, size, numEventsInWaitList,
                                           &phEventWaitList, phEvent));
}

TEST_P(urEnqueueReadHostPipeTest, InvalidNullPointerBuffer) {
    bool blocking = true;
    uint32_t numEventsInWaitList = 0;
    ur_event_handle_t phEventWaitList;
    ur_event_handle_t *phEvent = nullptr;

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueReadHostPipe(
                         queue, program, pipe_symbol, blocking, nullptr, size,
                         numEventsInWaitList, &phEventWaitList, phEvent));
}

TEST_P(urEnqueueReadHostPipeTest, InvalidEventWaitList) {
    bool blocking = true;
    ur_event_handle_t phEventWaitList;
    ur_event_handle_t *phEvent = nullptr;

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                     urEnqueueReadHostPipe(queue, program, pipe_symbol,
                                           blocking, &buffer, size, 1, nullptr,
                                           phEvent));

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                     urEnqueueReadHostPipe(queue, program, pipe_symbol,
                                           blocking, &buffer, size, 0,
                                           &phEventWaitList, phEvent));

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                     urEnqueueReadHostPipe(queue, program, pipe_symbol,
                                           blocking, &buffer, size, 0,
                                           &validEvent, nullptr));
}
