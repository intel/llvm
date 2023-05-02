// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urEnqueueWriteHostPipeTest = uur::urHostPipeTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueWriteHostPipeTest);

TEST_P(urEnqueueWriteHostPipeTest, InvalidNullHandleQueue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueWriteHostPipe(
                         nullptr, program, pipe_symbol, blocking, &buffer, size,
                         numEventsInWaitList, &phEventWaitList, phEvent));
}

TEST_P(urEnqueueWriteHostPipeTest, InvalidNullHandleProgram) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueWriteHostPipe(
                         queue, nullptr, pipe_symbol, blocking, &buffer, size,
                         numEventsInWaitList, &phEventWaitList, phEvent));
}

TEST_P(urEnqueueWriteHostPipeTest, InvalidNullPointerPipeSymbol) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueWriteHostPipe(queue, program, nullptr, blocking,
                                            &buffer, size, numEventsInWaitList,
                                            &phEventWaitList, phEvent));
}

TEST_P(urEnqueueWriteHostPipeTest, InvalidNullPointerBuffer) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueWriteHostPipe(
                         queue, program, pipe_symbol, blocking, nullptr, size,
                         numEventsInWaitList, &phEventWaitList, phEvent));
}

TEST_P(urEnqueueWriteHostPipeTest, InvalidNullPointerEvent) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueWriteHostPipe(
                         queue, program, pipe_symbol, blocking, &buffer, size,
                         numEventsInWaitList, &phEventWaitList, phEvent));
}

TEST_P(urEnqueueWriteHostPipeTest, InvalidEventWaitList) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                     urEnqueueWriteHostPipe(queue, program, pipe_symbol,
                                            blocking, &buffer, size, 1, nullptr,
                                            phEvent));

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                     urEnqueueWriteHostPipe(queue, program, pipe_symbol,
                                            blocking, &buffer, size, 0,
                                            &phEventWaitList, phEvent));

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                     urEnqueueWriteHostPipe(queue, program, pipe_symbol,
                                            blocking, &buffer, size, 0,
                                            &validEvent, nullptr));
}
