// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

using urDeviceRetainTest = uur::urAllDevicesTest;

TEST_F(urDeviceRetainTest, Success) {
  for (auto device : devices) {
    ASSERT_SUCCESS(urDeviceRetain(device));
    EXPECT_SUCCESS(urDeviceRelease(device));
  }
}

TEST_F(urDeviceRetainTest, InvalidNullHandle) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urDeviceRetain(nullptr));
}
