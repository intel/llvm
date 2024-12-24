// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ze_api.h"
#include <functional>
#include <memory>
#include <ur_api.h>
#include <uur/fixtures.h>

std::unique_ptr<_ze_event_handle_t, std::function<void(ze_event_handle_t)>>
createZeEvent(ur_context_handle_t hContext, ur_device_handle_t hDevice) {
    ze_event_pool_desc_t desc;
    desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    desc.pNext = nullptr;
    desc.count = 1;
    desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;

    ur_native_handle_t nativeContext;
    EXPECT_SUCCESS(urContextGetNativeHandle(hContext, &nativeContext));

    ur_native_handle_t nativeDevice;
    EXPECT_SUCCESS(urDeviceGetNativeHandle(hDevice, &nativeDevice));

    ze_event_pool_handle_t pool = nullptr;

    EXPECT_EQ(zeEventPoolCreate((ze_context_handle_t)nativeContext, &desc, 1,
                                (ze_device_handle_t *)&nativeDevice, &pool),
              ZE_RESULT_SUCCESS);

    ze_event_desc_t eventDesc;
    eventDesc.pNext = nullptr;
    eventDesc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    eventDesc.index = 0;
    eventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    eventDesc.wait = 0;

    ze_event_handle_t zeEvent;
    EXPECT_EQ(zeEventCreate(pool, &eventDesc, &zeEvent), ZE_RESULT_SUCCESS);

    return std::unique_ptr<_ze_event_handle_t,
                           std::function<void(ze_event_handle_t)>>(
        zeEvent, [pool](ze_event_handle_t zeEvent) {
            zeEventDestroy(zeEvent);
            zeEventPoolDestroy(pool);
        });
}