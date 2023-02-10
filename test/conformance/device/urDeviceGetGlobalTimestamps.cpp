// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urDeviceGetGlobalTimestampTest = uur::urAllDevicesTest;

TEST_F(urDeviceGetGlobalTimestampTest, Success) {
    for (auto device : devices) {
        uint64_t device_time = 0;
        uint64_t host_time = 0;
        ASSERT_SUCCESS(urDeviceGetGlobalTimestamps(device, &device_time, &host_time));
        ASSERT_NE(device_time, 0);
        ASSERT_NE(host_time, 0);
        // TODO - ASSERT synchronized? - #166
    }
}

TEST_F(urDeviceGetGlobalTimestampTest, SuccessHostTimer) {
    for (auto device : devices) {
        uint64_t host_time = 0;
        ASSERT_SUCCESS(urDeviceGetGlobalTimestamps(device, nullptr, &host_time));
        ASSERT_NE(host_time, 0);
    }
}

TEST_F(urDeviceGetGlobalTimestampTest, SuccessNoTimers) {
    for (auto device : devices) {
        ASSERT_SUCCESS(urDeviceGetGlobalTimestamps(device, nullptr, nullptr));
    }
}

TEST_F(urDeviceGetGlobalTimestampTest, InvalidNullHandleDevice) {
    uint64_t device_time = 0;
    uint64_t host_time = 0;
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urDeviceGetGlobalTimestamps(nullptr, &device_time, &host_time));
}
