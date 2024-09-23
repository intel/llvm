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

using namespace std::chrono_literals;
using urLevelZeroEventNativeHandleTest = uur::urQueueTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urLevelZeroEventNativeHandleTest);

#define TEST_MEMCPY_SIZE 4096

TEST_P(urLevelZeroEventNativeHandleTest, WaitForNative) {
    ze_event_pool_desc_t desc;
    desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    desc.pNext = nullptr;
    desc.count = 1;
    desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;

    ur_native_handle_t nativeContext;
    ASSERT_SUCCESS(urContextGetNativeHandle(context, &nativeContext));

    ur_native_handle_t nativeDevice;
    ASSERT_SUCCESS(urDeviceGetNativeHandle(device, &nativeDevice));

    ze_event_pool_handle_t pool = nullptr;

    ASSERT_EQ(zeEventPoolCreate((ze_context_handle_t)nativeContext, &desc, 1,
                                (ze_device_handle_t *)&nativeDevice, &pool),
              ZE_RESULT_SUCCESS);

    ze_event_desc_t eventDesc;
    eventDesc.pNext = nullptr;
    eventDesc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    eventDesc.index = 0;
    eventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    eventDesc.wait = 0;

    ze_event_handle_t zeEvent;
    ASSERT_EQ(zeEventCreate(pool, &eventDesc, &zeEvent), ZE_RESULT_SUCCESS);

    ur_event_native_properties_t pprops;
    pprops.isNativeHandleOwned = false;
    pprops.pNext = nullptr;
    pprops.stype = UR_STRUCTURE_TYPE_EVENT_NATIVE_PROPERTIES;

    ur_event_handle_t urEvent;
    ASSERT_SUCCESS(urEventCreateWithNativeHandle((ur_native_handle_t)zeEvent,
                                                 context, &pprops, &urEvent));

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

    zeEventHostSignal(zeEvent);

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
    zeEventDestroy(zeEvent);
    zeEventPoolDestroy(pool);
}
