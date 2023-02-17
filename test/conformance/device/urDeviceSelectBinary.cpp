// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urDeviceSelectBinaryTest = uur::urAllDevicesTest;

// TODO - Replace with valid UR binaries. - See #171
static const char *binaries[] = {"binary A", "binary B"};
static const uint32_t binaries_length = 2;

TEST_F(urDeviceSelectBinaryTest, Success) {
    for (auto device : devices) {
        uint32_t selected_binary = binaries_length; // invalid index
        ASSERT_SUCCESS(urDeviceSelectBinary(device, (const uint8_t **)binaries,
                                            binaries_length, &selected_binary));
        ASSERT_LT(selected_binary, 2);
    }
}

TEST_F(urDeviceSelectBinaryTest, InvalidNullHandleDevice) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urDeviceSelectBinary(nullptr, (const uint8_t **)binaries,
                                          binaries_length, nullptr));
}

TEST_F(urDeviceSelectBinaryTest, InvalidNullPointerBinaries) {
    for (auto device : devices) {
        uint32_t selected_binary;
        ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                         urDeviceSelectBinary(device, nullptr, binaries_length,
                                              &selected_binary));
    }
}

TEST_F(urDeviceSelectBinaryTest, InvalidNullPointerSelectedBinary) {
    for (auto device : devices) {
        ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                         urDeviceSelectBinary(device,
                                              (const uint8_t **)binaries,
                                              binaries_length, nullptr));
    }
}

TEST_F(urDeviceSelectBinaryTest, InvalidValueNumBinaries) {
    for (auto device : devices) {
        uint32_t selected_binary;
        ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_VALUE,
                         urDeviceSelectBinary(device,
                                              (const uint8_t **)binaries, 0,
                                              &selected_binary));
    }
}
