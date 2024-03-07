// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "helpers.h"
#include <uur/fixtures.h>

struct urEnqueueMemUnmapTestWithParam
    : uur::urMemBufferQueueTestWithParam<uur::mem_buffer_test_parameters_t> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            uur::urMemBufferQueueTestWithParam<
                uur::mem_buffer_test_parameters_t>::SetUp());
        ASSERT_SUCCESS(urEnqueueMemBufferMap(
            queue, buffer, true, UR_MAP_FLAG_READ | UR_MAP_FLAG_WRITE, 0, size,
            0, nullptr, nullptr, (void **)&map));
    };

    void TearDown() override {
        uur::urMemBufferQueueTestWithParam<
            uur::mem_buffer_test_parameters_t>::TearDown();
    }

    uint32_t *map = nullptr;
};

static std::vector<uur::mem_buffer_test_parameters_t> test_parameters{
    {1024, UR_MEM_FLAG_READ_WRITE},
    {2500, UR_MEM_FLAG_READ_WRITE},
    {4096, UR_MEM_FLAG_READ_WRITE},
    {6000, UR_MEM_FLAG_READ_WRITE},
    {1024, UR_MEM_FLAG_WRITE_ONLY},
    {2500, UR_MEM_FLAG_WRITE_ONLY},
    {4096, UR_MEM_FLAG_WRITE_ONLY},
    {6000, UR_MEM_FLAG_WRITE_ONLY},
    {1024, UR_MEM_FLAG_READ_ONLY},
    {2500, UR_MEM_FLAG_READ_ONLY},
    {4096, UR_MEM_FLAG_READ_ONLY},
    {6000, UR_MEM_FLAG_READ_ONLY},
    {1024, UR_MEM_FLAG_ALLOC_HOST_POINTER},
    {2500, UR_MEM_FLAG_ALLOC_HOST_POINTER},
    {4096, UR_MEM_FLAG_ALLOC_HOST_POINTER},
    {6000, UR_MEM_FLAG_ALLOC_HOST_POINTER},
};

UUR_TEST_SUITE_P(urEnqueueMemUnmapTestWithParam,
                 ::testing::ValuesIn(test_parameters),
                 uur::printMemBufferTestString<urEnqueueMemUnmapTestWithParam>);

TEST_P(urEnqueueMemUnmapTestWithParam, Success) {
    ASSERT_SUCCESS(urEnqueueMemUnmap(queue, buffer, map, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemUnmapTestWithParam, InvalidNullHandleQueue) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urEnqueueMemUnmap(nullptr, buffer, map, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemUnmapTestWithParam, InvalidNullHandleMem) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urEnqueueMemUnmap(queue, nullptr, map, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemUnmapTestWithParam, InvalidNullPtrMap) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_POINTER,
        urEnqueueMemUnmap(queue, buffer, nullptr, 0, nullptr, nullptr));
}

TEST_P(urEnqueueMemUnmapTestWithParam, InvalidNullPtrEventWaitList) {
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
