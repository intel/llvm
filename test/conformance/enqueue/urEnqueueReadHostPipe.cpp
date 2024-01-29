// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urEnqueueReadHostPipeTest = uur::urHostPipeTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueReadHostPipeTest);

TEST_P(urEnqueueReadHostPipeTest, InvalidNullHandleQueue) {
    uint32_t numEventsInWaitList = 0;
    ur_event_handle_t phEventWaitList;
    ur_event_handle_t *phEvent = nullptr;

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueReadHostPipe(nullptr, program, pipe_symbol,
                                           /*blocking*/ true, &buffer, size,
                                           numEventsInWaitList,
                                           &phEventWaitList, phEvent));
}

TEST_P(urEnqueueReadHostPipeTest, InvalidNullHandleProgram) {
    uint32_t numEventsInWaitList = 0;
    ur_event_handle_t phEventWaitList;
    ur_event_handle_t *phEvent = nullptr;

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueReadHostPipe(queue, nullptr, pipe_symbol,
                                           /*blocking*/ true, &buffer, size,
                                           numEventsInWaitList,
                                           &phEventWaitList, phEvent));
}

TEST_P(urEnqueueReadHostPipeTest, InvalidNullPointerPipeSymbol) {
    uint32_t numEventsInWaitList = 0;
    ur_event_handle_t phEventWaitList;
    ur_event_handle_t *phEvent = nullptr;

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueReadHostPipe(
                         queue, program, nullptr, /*blocking*/ true, &buffer,
                         size, numEventsInWaitList, &phEventWaitList, phEvent));
}

TEST_P(urEnqueueReadHostPipeTest, InvalidNullPointerBuffer) {
    uint32_t numEventsInWaitList = 0;
    ur_event_handle_t phEventWaitList;
    ur_event_handle_t *phEvent = nullptr;

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urEnqueueReadHostPipe(queue, program, pipe_symbol,
                                           /*blocking*/ true, nullptr, size,
                                           numEventsInWaitList,
                                           &phEventWaitList, phEvent));
}

TEST_P(urEnqueueReadHostPipeTest, InvalidEventWaitList) {
    ur_event_handle_t phEventWaitList;
    ur_event_handle_t *phEvent = nullptr;

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                     urEnqueueReadHostPipe(queue, program, pipe_symbol,
                                           /*blocking*/ true, &buffer, size, 1,
                                           nullptr, phEvent));

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                     urEnqueueReadHostPipe(queue, program, pipe_symbol,
                                           /*blocking*/ true, &buffer, size, 0,
                                           &phEventWaitList, phEvent));

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST,
                     urEnqueueReadHostPipe(queue, program, pipe_symbol,
                                           /*blocking*/ true, &buffer, size, 0,
                                           &validEvent, nullptr));

    ur_event_handle_t inv_evt = nullptr;
    ASSERT_EQ_RESULT(urEnqueueReadHostPipe(queue, program, pipe_symbol,
                                           /*blocking*/ true, &buffer, size, 1,
                                           &inv_evt, nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ASSERT_SUCCESS(urEventRelease(validEvent));
}
