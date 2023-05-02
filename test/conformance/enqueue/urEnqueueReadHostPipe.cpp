// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urEnqueueReadHostPipeTest = uur::urHostPipeTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueReadHostPipeTest);

TEST_P(urEnqueueReadHostPipeTest, InvalidNullHandleQueue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueReadHostPipe(
                         nullptr, program, pipe_symbol, blocking, &buffer, size,
                         numEventsInWaitList, &phEventWaitList, phEvent));
}

TEST_P(urEnqueueReadHostPipeTest, InvalidNullHandleProgram) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueReadHostPipe(
                         queue, nullptr, pipe_symbol, blocking, &buffer, size,
                         numEventsInWaitList, &phEventWaitList, phEvent));
}

TEST_P(urEnqueueReadHostPipeTest, InvalidNullPointerPipeSymbol) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueReadHostPipe(queue, program, nullptr, blocking,
                                           &buffer, size, numEventsInWaitList,
                                           &phEventWaitList, phEvent));
}

TEST_P(urEnqueueReadHostPipeTest, InvalidNullPointerBuffer) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueReadHostPipe(
                         queue, program, pipe_symbol, blocking, nullptr, size,
                         numEventsInWaitList, &phEventWaitList, phEvent));
}

TEST_P(urEnqueueReadHostPipeTest, InvalidEventWaitList) {
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
