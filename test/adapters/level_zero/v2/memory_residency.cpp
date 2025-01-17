// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include "uur/fixtures.h"
#include "uur/utils.h"

using urMemoryResidencyTest = uur::urMultiDeviceContextTestTemplate<1>;
UUR_INSTANTIATE_PLATFORM_TEST_SUITE_P(urMemoryResidencyTest);

TEST_P(urMemoryResidencyTest, allocatingDeviceMemoryWillResultInOOM) {
  static constexpr size_t allocSize = 1024 * 1024;

  if (!uur::isPVC(devices[0])) {
    GTEST_SKIP() << "Test requires a PVC device";
  }

  size_t initialMemFree = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(devices[0], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                                 sizeof(size_t), &initialMemFree, nullptr));

  if (initialMemFree < allocSize) {
    GTEST_SKIP() << "Not enough device memory available";
  }

  void *ptr = nullptr;
  ASSERT_SUCCESS(
      urUSMDeviceAlloc(context, devices[0], nullptr, nullptr, allocSize, &ptr));

  size_t currentMemFree = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(devices[0], UR_DEVICE_INFO_GLOBAL_MEM_FREE,
                                 sizeof(size_t), &currentMemFree, nullptr));

  // amount of free memory should decrease after making a memory allocation
  // resident
  ASSERT_LE(currentMemFree, initialMemFree);

  ASSERT_SUCCESS(urUSMFree(context, ptr));
}
