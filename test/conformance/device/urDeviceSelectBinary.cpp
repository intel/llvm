// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>
using urDeviceSelectBinaryTest = uur::urAllDevicesTest;

static constexpr ur_device_binary_t binaries[] = {
    {UR_STRUCTURE_TYPE_DEVICE_BINARY, nullptr, UR_DEVICE_BINARY_TARGET_UNKNOWN},
    {UR_STRUCTURE_TYPE_DEVICE_BINARY, nullptr, UR_DEVICE_BINARY_TARGET_SPIRV32},
    {UR_STRUCTURE_TYPE_DEVICE_BINARY, nullptr, UR_DEVICE_BINARY_TARGET_SPIRV64},
    {UR_STRUCTURE_TYPE_DEVICE_BINARY, nullptr,
     UR_DEVICE_BINARY_TARGET_SPIRV64_X86_64},
    {UR_STRUCTURE_TYPE_DEVICE_BINARY, nullptr,
     UR_DEVICE_BINARY_TARGET_SPIRV64_GEN},
    {UR_STRUCTURE_TYPE_DEVICE_BINARY, nullptr,
     UR_DEVICE_BINARY_TARGET_SPIRV64_FPGA},
    {UR_STRUCTURE_TYPE_DEVICE_BINARY, nullptr, UR_DEVICE_BINARY_TARGET_NVPTX64},
    {UR_STRUCTURE_TYPE_DEVICE_BINARY, nullptr, UR_DEVICE_BINARY_TARGET_AMDGCN}};
static constexpr uint32_t binaries_length =
    sizeof(binaries) / sizeof(ur_device_binary_t);

TEST_F(urDeviceSelectBinaryTest, Success) {
    for (auto device : devices) {
        uint32_t selected_binary = binaries_length; // invalid index
        ASSERT_SUCCESS(urDeviceSelectBinary(device, binaries, binaries_length,
                                            &selected_binary));
        ASSERT_LT(selected_binary, binaries_length);
    }
}

TEST_F(urDeviceSelectBinaryTest, InvalidNullHandleDevice) {
    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_NULL_HANDLE,
        urDeviceSelectBinary(nullptr, binaries, binaries_length, nullptr));
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
        ASSERT_EQ_RESULT(
            UR_RESULT_ERROR_INVALID_NULL_POINTER,
            urDeviceSelectBinary(device, binaries, binaries_length, nullptr));
    }
}

TEST_F(urDeviceSelectBinaryTest, InvalidValueNumBinaries) {
    for (auto device : devices) {
        uint32_t selected_binary;
        ASSERT_EQ_RESULT(
            UR_RESULT_ERROR_INVALID_VALUE,
            urDeviceSelectBinary(device, binaries, 0, &selected_binary));
    }
}
