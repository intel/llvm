// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "helpers.h"
#include "uur/utils.h"
#include <uur/fixtures.h>
#include <uur/known_failure.h>

#include <umf.h>
#include <umf/memory_pool.h>
#include <umf/memory_provider.h>

struct umfDeviceAllocTest : uur::urUSMAllocTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urUSMAllocTest::SetUp());
    ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, USMSupport));
    if (!USMSupport) {
      GTEST_SKIP() << "Device USM is not supported.";
    }
  }
};

// The 0 value parameters are not relevant for umfDeviceAllocTest tests, they
// are used below in urUSMDeviceAllocAlignmentTest for allocation size and
// alignment values
UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    umfDeviceAllocTest,
    testing::Combine(
        testing::ValuesIn(uur::BoolTestParam::makeBoolParam("UsePool")),
        testing::Values(0), testing::Values(0),
        testing::Values(UR_USM_ADVICE_FLAG_DEFAULT)),
    uur::printUSMAllocTestString<umfDeviceAllocTest>);

TEST_P(umfDeviceAllocTest, UMFAllocSuccessfull) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{}, uur::CUDA{}, uur::HIP{},
                       uur::OpenCL{});

  void *ptr = nullptr;
  size_t allocation_size = sizeof(int);
  ASSERT_SUCCESS(
      urUSMDeviceAlloc(context, device, nullptr, pool, allocation_size, &ptr));
  ASSERT_NE(ptr, nullptr);

  auto umfPool = umfPoolByPtr(ptr);
  ASSERT_NE(umfPool, nullptr);

  umf_memory_provider_handle_t hProvider;
  ASSERT_EQ(umfPoolGetMemoryProvider(umfPool, &hProvider), UMF_RESULT_SUCCESS);
  ASSERT_NE(hProvider, nullptr);

  // make sure that pool can be used for allocations
  void *umfPtr = umfPoolMalloc(umfPool, allocation_size);
  ASSERT_NE(umfPtr, nullptr);
  ASSERT_EQ(umfPoolFree(umfPool, umfPtr), UMF_RESULT_SUCCESS);

  ASSERT_SUCCESS(urUSMFree(context, ptr));
}
