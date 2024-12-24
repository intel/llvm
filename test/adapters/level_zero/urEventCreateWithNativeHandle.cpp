// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ur_api.h"
#include "uur/checks.h"
#include "ze_api.h"
#include <cstring>
#include <thread>
#include <uur/fixtures.h>

#include "ze_helpers.hpp"

using namespace std::chrono_literals;
using urLevelZeroEventNativeHandleTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urLevelZeroEventNativeHandleTest);

#define TEST_MEMCPY_SIZE 4096

TEST_P(urLevelZeroEventNativeHandleTest, WaitForNative) {
    auto zeEvent = createZeEvent(context, device);

    ur_event_native_properties_t pprops;
    pprops.isNativeHandleOwned = false;
    pprops.pNext = nullptr;
    pprops.stype = UR_STRUCTURE_TYPE_EVENT_NATIVE_PROPERTIES;

    ur_event_handle_t urEvent;
    ASSERT_SUCCESS(urEventCreateWithNativeHandle(
        (ur_native_handle_t)zeEvent.get(), context, &pprops, &urEvent));

    int *src = (int *)malloc(TEST_MEMCPY_SIZE);
    memset(src, 0xc, TEST_MEMCPY_SIZE);

    int *dst = (int *)malloc(TEST_MEMCPY_SIZE);
    memset(dst, 0, TEST_MEMCPY_SIZE);

    int *dst2 = (int *)malloc(TEST_MEMCPY_SIZE);
    memset(dst, 0, TEST_MEMCPY_SIZE);

    ur_event_handle_t memcpyEvent2;
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, dst2, src, TEST_MEMCPY_SIZE,
                                      0, nullptr, &memcpyEvent2));

    ur_event_handle_t memcpyEvent3;
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, dst2, src, TEST_MEMCPY_SIZE,
                                      0, nullptr, &memcpyEvent3));

    // just to make wait lists contain more than 1 event
    ur_event_handle_t events[] = {memcpyEvent2, urEvent, memcpyEvent3};

    ur_event_handle_t waitEvent;
    ASSERT_SUCCESS(
        urEnqueueEventsWaitWithBarrier(queue, 3, events, &waitEvent));

    ur_event_handle_t memcpyEvent;
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, false, dst, src, TEST_MEMCPY_SIZE,
                                      1, &waitEvent, &memcpyEvent));

    // urQueueFinish would hang, so we flush and then wait
    // some time to make sure the gpu had plenty of time
    // to do the memcpy.
    urQueueFlush(queue);
    std::this_thread::sleep_for(500ms);

    ASSERT_NE(memcmp(src, dst, TEST_MEMCPY_SIZE), 0);

    zeEventHostSignal(zeEvent.get());

    urQueueFinish(queue);

    ASSERT_EQ(memcmp(src, dst, 4096), 0);

    free(src);
    free(dst);
    free(dst2);
    urEventRelease(urEvent);
    urEventRelease(waitEvent);
    urEventRelease(memcpyEvent);
    urEventRelease(memcpyEvent2);
    urEventRelease(memcpyEvent3);
}

TEST_P(urLevelZeroEventNativeHandleTest, NativeStatusQuery) {
    auto zeEvent = createZeEvent(context, device);

    ur_event_native_properties_t pprops;
    pprops.isNativeHandleOwned = false;
    pprops.pNext = nullptr;
    pprops.stype = UR_STRUCTURE_TYPE_EVENT_NATIVE_PROPERTIES;

    ur_event_handle_t urEvent;
    ASSERT_SUCCESS(urEventCreateWithNativeHandle(
        (ur_native_handle_t)zeEvent.get(), context, &pprops, &urEvent));

    ur_event_status_t status;
    ASSERT_SUCCESS(urEventGetInfo(urEvent,
                                  UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                                  sizeof(ur_event_status_t), &status, nullptr));
    ASSERT_EQ(status, UR_EVENT_STATUS_SUBMITTED);

    zeEventHostSignal(zeEvent.get());

    ASSERT_SUCCESS(urEventGetInfo(urEvent,
                                  UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                                  sizeof(ur_event_status_t), &status, nullptr));
    ASSERT_EQ(status, UR_EVENT_STATUS_COMPLETE);

    urEventRelease(urEvent);
}
