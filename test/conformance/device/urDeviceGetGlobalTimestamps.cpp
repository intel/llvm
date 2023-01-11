// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urDeviceGetGlobalTimestampTest = uur::urAllDevicesTest;

TEST_F(urDeviceGetGlobalTimestampTest, DISABLED_Success) {
  for (auto device : devices) {
    uint64_t device_time = 0;
    uint64_t host_time = 0;
    ASSERT_SUCCESS(
        urDeviceGetGlobalTimestamps(device, &device_time, &host_time));
    ASSERT_NE(device_time, 0);
    ASSERT_NE(host_time, 0);
    // TODO - ASSERT synchronized? - #166
  }
}

TEST_F(urDeviceGetGlobalTimestampTest, DISABLED_SuccessHostTimer) {
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
