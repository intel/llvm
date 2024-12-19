// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ur_print.hpp"
#include "uur/fixtures.h"
#include "uur/raii.h"

#include <map>
#include <string>

#include "ze_tracer_common.hpp"

size_t zeCommandListAppendWaitOnEventsCount = 0;

void OnAppendWaitOnEventsCb(ze_command_list_append_wait_on_events_params_t *,
                            ze_result_t, void *, void **) {
    zeCommandListAppendWaitOnEventsCount++;
}

static std::shared_ptr<_zel_tracer_handle_t> tracer = [] {
    zel_core_callbacks_t prologue_callbacks{};
    prologue_callbacks.CommandList.pfnAppendWaitOnEventsCb =
        OnAppendWaitOnEventsCb;
    return enableTracing(prologue_callbacks, {});
}();

using urMultiQueueMultiDeviceEventCacheTest = uur::urAllDevicesTest;
TEST_F(urMultiQueueMultiDeviceEventCacheTest,
       GivenMultiSubDeviceWithQueuePerSubDeviceThenEventIsSharedBetweenQueues) {
    uint32_t max_sub_devices = 0;
    ASSERT_SUCCESS(
        uur::GetDevicePartitionMaxSubDevices(devices[0], max_sub_devices));
    if (max_sub_devices < 2) {
        GTEST_SKIP();
    }
    ur_device_partition_property_t prop;
    prop.type = UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
    prop.value.affinity_domain =
        UR_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE;

    ur_device_partition_properties_t properties{
        UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES,
        nullptr,
        &prop,
        1,
    };
    uint32_t numSubDevices = 0;
    ASSERT_SUCCESS(
        urDevicePartition(devices[0], &properties, 0, nullptr, &numSubDevices));
    std::vector<ur_device_handle_t> sub_devices;
    sub_devices.reserve(numSubDevices);
    ASSERT_SUCCESS(urDevicePartition(devices[0], &properties, numSubDevices,
                                     sub_devices.data(), nullptr));
    uur::raii::Context context1 = nullptr;
    ASSERT_SUCCESS(urContextCreate(sub_devices.size(), &sub_devices[0], nullptr,
                                   context1.ptr()));
    ASSERT_NE(nullptr, context1);
    uur::raii::Context context2 = nullptr;
    ASSERT_SUCCESS(urContextCreate(sub_devices.size(), &sub_devices[0], nullptr,
                                   context2.ptr()));
    ASSERT_NE(nullptr, context2);
    ur_queue_handle_t queue1 = nullptr;
    ASSERT_SUCCESS(urQueueCreate(context1, sub_devices[0], 0, &queue1));
    ur_queue_handle_t queue2 = nullptr;
    ASSERT_SUCCESS(urQueueCreate(context2, sub_devices[1], 0, &queue2));
    uur::raii::Event event = nullptr;
    uur::raii::Event eventWait = nullptr;
    uur::raii::Event eventWaitDummy = nullptr;
    zeCommandListAppendWaitOnEventsCount = 0;
    EXPECT_SUCCESS(urEventCreateWithNativeHandle(0, context1, nullptr,
                                                 eventWaitDummy.ptr()));
    EXPECT_SUCCESS(
        urEnqueueEventsWait(queue1, 1, eventWaitDummy.ptr(), eventWait.ptr()));
    EXPECT_SUCCESS(
        urEnqueueEventsWait(queue2, 1, eventWait.ptr(), event.ptr()));
    EXPECT_EQ(zeCommandListAppendWaitOnEventsCount, 2);
    ASSERT_SUCCESS(urEventRelease(eventWaitDummy.get()));
    ASSERT_SUCCESS(urEventRelease(eventWait.get()));
    ASSERT_SUCCESS(urEventRelease(event.get()));
    ASSERT_SUCCESS(urQueueRelease(queue2));
    ASSERT_SUCCESS(urQueueRelease(queue1));
}

TEST_F(urMultiQueueMultiDeviceEventCacheTest,
       GivenMultiDeviceWithQueuePerDeviceThenMultiDeviceEventIsCreated) {
    if (devices.size() < 2) {
        GTEST_SKIP();
    }
    uur::raii::Context context1 = nullptr;
    ASSERT_SUCCESS(
        urContextCreate(devices.size(), &devices[0], nullptr, context1.ptr()));
    ASSERT_NE(nullptr, context1);
    uur::raii::Context context2 = nullptr;
    ASSERT_SUCCESS(
        urContextCreate(devices.size(), &devices[0], nullptr, context2.ptr()));
    ASSERT_NE(nullptr, context2);
    ur_queue_handle_t queue1 = nullptr;
    ASSERT_SUCCESS(urQueueCreate(context1, devices[0], 0, &queue1));
    ur_queue_handle_t queue2 = nullptr;
    ASSERT_SUCCESS(urQueueCreate(context2, devices[1], 0, &queue2));
    uur::raii::Event event = nullptr;
    uur::raii::Event eventWait = nullptr;
    uur::raii::Event eventWaitDummy = nullptr;
    zeCommandListAppendWaitOnEventsCount = 0;
    EXPECT_SUCCESS(urEventCreateWithNativeHandle(0, context1, nullptr,
                                                 eventWaitDummy.ptr()));
    EXPECT_SUCCESS(
        urEnqueueEventsWait(queue1, 1, eventWaitDummy.ptr(), eventWait.ptr()));
    EXPECT_SUCCESS(
        urEnqueueEventsWait(queue2, 1, eventWait.ptr(), event.ptr()));
    EXPECT_EQ(zeCommandListAppendWaitOnEventsCount, 2);
    ASSERT_SUCCESS(urEventRelease(eventWaitDummy.get()));
    ASSERT_SUCCESS(urEventRelease(eventWait.get()));
    ASSERT_SUCCESS(urEventRelease(event.get()));
    ASSERT_SUCCESS(urQueueRelease(queue2));
    ASSERT_SUCCESS(urQueueRelease(queue1));
}
