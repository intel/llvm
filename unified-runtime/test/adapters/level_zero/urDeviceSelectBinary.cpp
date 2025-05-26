// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ur_api.h"
#include <uur/fixtures.h>

#include <array>
#include <vector>

using urLevelZeroDeviceSelectBinaryTest = uur::urDeviceTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urLevelZeroDeviceSelectBinaryTest);

static ur_device_binary_t binary_for_tgt(const char *Target) {
  return {UR_STRUCTURE_TYPE_DEVICE_BINARY, nullptr, Target};
}

TEST_P(urLevelZeroDeviceSelectBinaryTest, TargetPreference) {
  std::vector<ur_device_binary_t> binaries = {
      binary_for_tgt(UR_DEVICE_BINARY_TARGET_UNKNOWN),
      binary_for_tgt(UR_DEVICE_BINARY_TARGET_SPIRV64),
      binary_for_tgt(UR_DEVICE_BINARY_TARGET_SPIRV64_GEN)};

  // Gen binary should be preferred over SPIR-V
  {
    uint32_t selected_binary = binaries.size(); // invalid index
    ASSERT_SUCCESS(urDeviceSelectBinary(device, binaries.data(),
                                        binaries.size(), &selected_binary));
    ASSERT_EQ(selected_binary, binaries.size() - 1);
  }

  // Remove the Gen binary,
  // SPIR-V should be selected
  binaries.pop_back();
  {
    uint32_t selected_binary = binaries.size(); // invalid index
    ASSERT_SUCCESS(urDeviceSelectBinary(device, binaries.data(),
                                        binaries.size(), &selected_binary));
    ASSERT_EQ(selected_binary, binaries.size() - 1);
  }

  // No supported binaries left, should return an error
  binaries.pop_back();
  {
    uint32_t selected_binary = binaries.size(); // invalid index
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_BINARY,
                     urDeviceSelectBinary(device, binaries.data(),
                                          binaries.size(), &selected_binary));
  }
}

TEST_P(urLevelZeroDeviceSelectBinaryTest, FirstOfSupported) {
  std::vector<const char *> SupportedTargets = {
      UR_DEVICE_BINARY_TARGET_SPIRV64,
      UR_DEVICE_BINARY_TARGET_SPIRV64_GEN,
  };
  for (const char *Target : SupportedTargets) {
    std::array binaries = {
        binary_for_tgt(UR_DEVICE_BINARY_TARGET_UNKNOWN),
        binary_for_tgt(Target),
        binary_for_tgt(UR_DEVICE_BINARY_TARGET_AMDGCN),
        binary_for_tgt(Target),
    };

    uint32_t selected_binary = binaries.size(); // invalid index
    ASSERT_SUCCESS(urDeviceSelectBinary(device, binaries.data(),
                                        binaries.size(), &selected_binary));
    ASSERT_EQ(selected_binary, 1u);
  }
}
