// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %with-v2 ZES_ENABLE_SYSMAN=1 ./memory_residency-test
// REQUIRES: v2

#include "uur/fixtures.h"
#include "uur/utils.h"

using urMemoryResidencyTest = uur::urMultiDeviceContextTestTemplate<1>;
UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urMemoryResidencyTest);

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

struct urMemoryMultiResidencyTest : uur::urMultiDeviceContextTestTemplate<2> {

  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urMultiDeviceContextTestTemplate<2>::SetUp());

    for (std::size_t i = 0; i < 2; i++) {
      ur_bool_t usm_p2p_support = false;
      ASSERT_SUCCESS(
          urDeviceGetInfo(devices[i], UR_DEVICE_INFO_USM_P2P_SUPPORT_EXP,
                          sizeof(usm_p2p_support), &usm_p2p_support, nullptr));
      if (!usm_p2p_support) {
        GTEST_SKIP() << "EXP usm p2p feature is not supported.";
      }
    }
  }

  void TearDown() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urMultiDeviceContextTestTemplate<2>::TearDown());
  }
};

UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urMemoryMultiResidencyTest);

TEST_P(urMemoryMultiResidencyTest, allocationInitiallyAbsentOnPeer) {}

TEST_P(urMemoryMultiResidencyTest, allocationExistsOnPeerWithEnabledAccess) {

  void *ptr = nullptr;
  ASSERT_SUCCESS(
      urUSMDeviceAlloc(context, devices[0], nullptr, nullptr, 1, &ptr));
}

TEST_P(urMemoryMultiResidencyTest, allocationAbsentOnPeerWithDisabledAccess) {}