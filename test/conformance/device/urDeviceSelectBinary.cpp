// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urDeviceSelectBinaryTest = uur::urAllDevicesTest;

// TODO - This is just a placeholder test until we have a better understanding
// of the compiler requirements for UR. - See #171
TEST_F(urDeviceSelectBinaryTest, Success) {
  const char *binaries[] = {"binary A", "binary B"};
  uint32_t num_binaries = 2;
  for (auto device : devices) {
    ASSERT_SUCCESS(urDeviceSelectBinary(device, (const uint8_t **)binaries,
                                        num_binaries, nullptr));
  }
}

TEST_F(urDeviceSelectBinaryTest, InvalidNullHandle) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urDeviceSelectBinary(nullptr, nullptr, 0, nullptr));
}
