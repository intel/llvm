// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>
#include <uur/known_failure.h>

using urDeviceSelectBinaryTest = uur::urDeviceTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urDeviceSelectBinaryTest);

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

TEST_P(urDeviceSelectBinaryTest, Success) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  uint32_t selected_binary = binaries_length; // invalid index
  ASSERT_SUCCESS(urDeviceSelectBinary(device, binaries, binaries_length,
                                      &selected_binary));
  ASSERT_LT(selected_binary, binaries_length);
}

TEST_P(urDeviceSelectBinaryTest, InvalidNullHandleDevice) {
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_HANDLE,
      urDeviceSelectBinary(nullptr, binaries, binaries_length, nullptr));
}

TEST_P(urDeviceSelectBinaryTest, InvalidNullPointerBinaries) {
  uint32_t selected_binary;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_POINTER,
      urDeviceSelectBinary(device, nullptr, binaries_length, &selected_binary));
}

TEST_P(urDeviceSelectBinaryTest, InvalidNullPointerSelectedBinary) {
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_POINTER,
      urDeviceSelectBinary(device, binaries, binaries_length, nullptr));
}

TEST_P(urDeviceSelectBinaryTest, InvalidValueNumBinaries) {
  uint32_t selected_binary;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urDeviceSelectBinary(device, binaries, 0, &selected_binary));
}
