// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ze_api.h"

#include <uur/fixtures.h>
#include <vector>

using T = uint32_t;

struct urLevelZeroEnqueueNativeCommandTest : uur::urQueueTest {
    void SetUp() {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::SetUp());

        host_vec = std::vector<T>(global_size, 0);
        ASSERT_EQ(host_vec.size(), global_size);
        ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                        allocation_size, &device_ptr));
        ASSERT_NE(device_ptr, nullptr);
    }
    static constexpr T val = 42;
    static constexpr uint32_t global_size = 1e7;
    std::vector<T> host_vec;
    void *device_ptr = nullptr;
    static constexpr size_t allocation_size = sizeof(val) * global_size;
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urLevelZeroEnqueueNativeCommandTest);

struct InteropData1 {
    void *fill_ptr;
};

// Fill a device ptr with the pattern val
void interop_func_1(ur_queue_handle_t hQueue, void *data) {
    ze_command_list_handle_t CommandList;
    ASSERT_SUCCESS(urQueueGetNativeHandle(hQueue, nullptr,
                                          (ur_native_handle_t *)&CommandList));
    InteropData1 *func_data = reinterpret_cast<InteropData1 *>(data);

    // If L0 interop becomes a real use case we should make a new UR entry point
    // to propagate events into and out of the the interop func.
    zeCommandListAppendMemoryFill(
        CommandList, func_data->fill_ptr,
        &urLevelZeroEnqueueNativeCommandTest::val,
        sizeof(urLevelZeroEnqueueNativeCommandTest::val),
        urLevelZeroEnqueueNativeCommandTest::allocation_size, nullptr, 0,
        nullptr);
}

struct InteropData2 {
    void *from, *to;
};

// Read from device ptr to host ptr
void interop_func_2(ur_queue_handle_t hQueue, void *data) {
    ze_command_list_handle_t CommandList;
    ASSERT_SUCCESS(urQueueGetNativeHandle(hQueue, nullptr,
                                          (ur_native_handle_t *)&CommandList));
    InteropData2 *func_data = reinterpret_cast<InteropData2 *>(data);

    // If L0 interop becomes a real use case we should make a new UR entry point
    // to propagate events into and out of the the interop func.
    zeCommandListAppendMemoryCopy(
        CommandList, func_data->to, func_data->from,
        urLevelZeroEnqueueNativeCommandTest::allocation_size, nullptr, 0,
        nullptr);
}

TEST_P(urLevelZeroEnqueueNativeCommandTest, Success) {
    InteropData1 data_1{device_ptr};
    ur_event_handle_t event_1;
    ASSERT_SUCCESS(urEnqueueNativeCommandExp(
        queue, &interop_func_1, &data_1, 0, nullptr /*phMemList=*/,
        nullptr /*pProperties=*/, 0, nullptr /*phEventWaitList=*/, &event_1));
}

TEST_P(urLevelZeroEnqueueNativeCommandTest, Dependencies) {
    ur_event_handle_t event_1, event_2;

    InteropData1 data_1{device_ptr};
    ASSERT_SUCCESS(urEnqueueNativeCommandExp(
        queue, &interop_func_1, &data_1, 0, nullptr /*phMemList=*/,
        nullptr /*pProperties=*/, 0, nullptr /*phEventWaitList=*/, &event_1));

    InteropData2 data_2{device_ptr, host_vec.data()};
    ASSERT_SUCCESS(urEnqueueNativeCommandExp(
        queue, &interop_func_2, &data_2, 0, nullptr /*phMemList=*/,
        nullptr /*pProperties=*/, 1, &event_1, &event_2));
    urQueueFinish(queue);
    for (auto &i : host_vec) {
        ASSERT_EQ(i, val);
    }
}

TEST_P(urLevelZeroEnqueueNativeCommandTest, DependenciesURBefore) {
    ur_event_handle_t event_1, event_2;

    ASSERT_SUCCESS(urEnqueueUSMFill(queue, device_ptr, sizeof(val), &val,
                                    allocation_size, 0,
                                    nullptr /*phEventWaitList=*/, &event_1));

    InteropData2 data_2{device_ptr, host_vec.data()};
    ASSERT_SUCCESS(urEnqueueNativeCommandExp(
        queue, &interop_func_2, &data_2, 0, nullptr /*phMemList=*/,
        nullptr /*pProperties=*/, 1, &event_1, &event_2));
    urQueueFinish(queue);
    for (auto &i : host_vec) {
        ASSERT_EQ(i, val);
    }
}

TEST_P(urLevelZeroEnqueueNativeCommandTest, DependenciesURAfter) {
    ur_event_handle_t event_1;

    InteropData1 data_1{device_ptr};
    ASSERT_SUCCESS(urEnqueueNativeCommandExp(
        queue, &interop_func_1, &data_1, 0, nullptr /*phMemList=*/,
        nullptr /*pProperties=*/, 0, nullptr /*phEventWaitList=*/, &event_1));

    urEnqueueUSMMemcpy(queue, /*blocking*/ true, host_vec.data(), device_ptr,
                       allocation_size, 1, &event_1, nullptr);
    for (auto &i : host_vec) {
        ASSERT_EQ(i, val);
    }
}
