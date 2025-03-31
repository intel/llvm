// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/utils.h"
#include <uur/fixtures.h>

struct urUSMContextMemcpyExpTest : uur::urQueueTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());

    bool context_memcpy_support = false;
    ASSERT_SUCCESS(
        uur::GetUSMContextMemcpyExpSupport(device, context_memcpy_support));
    if (!context_memcpy_support) {
      GTEST_SKIP() << "urUSMContextMemcpyExp is not supported";
    }
  }

  void TearDown() override {
    if (src_ptr) {
      EXPECT_SUCCESS(urUSMFree(context, src_ptr));
    }
    if (dst_ptr) {
      EXPECT_SUCCESS(urUSMFree(context, dst_ptr));
    }

    UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::TearDown());
  }

  void initAllocations() {
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, src_ptr, sizeof(memset_src_value),
                                    &memset_src_value, allocation_size, 0,
                                    nullptr, nullptr));
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, dst_ptr, sizeof(memset_dst_value),
                                    &memset_dst_value, allocation_size, 0,
                                    nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
  }

  void verifyData() {
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, true, &host_mem, dst_ptr,
                                      allocation_size, 0, nullptr, nullptr));
    ASSERT_EQ(host_mem, memset_src_value);
  }

  static constexpr size_t memset_src_value = 42;
  static constexpr uint8_t memset_dst_value = 0;
  static constexpr uint32_t allocation_size = sizeof(memset_src_value);
  size_t host_mem = 0;

  void *src_ptr{nullptr};
  void *dst_ptr{nullptr};
};

struct urUSMContextMemcpyExpTestDevice : urUSMContextMemcpyExpTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urUSMContextMemcpyExpTest::SetUp());

    ur_device_usm_access_capability_flags_t device_usm = 0;
    ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, device_usm));
    if (!device_usm) {
      GTEST_SKIP() << "Device USM is not supported";
    }

    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                    allocation_size,
                                    reinterpret_cast<void **>(&src_ptr)));
    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                    allocation_size,
                                    reinterpret_cast<void **>(&dst_ptr)));
    initAllocations();
  }
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urUSMContextMemcpyExpTestDevice);

TEST_P(urUSMContextMemcpyExpTestDevice, Success) {
  ASSERT_SUCCESS(
      urUSMContextMemcpyExp(context, dst_ptr, src_ptr, allocation_size));
  verifyData();
}

// Arbitrarily do the negative tests with device allocations. These are mostly a
// test of the loader and validation layer anyway so no big deal if they don't
// run on all devices due to lack of support.
TEST_P(urUSMContextMemcpyExpTestDevice, InvalidNullContext) {
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_HANDLE,
      urUSMContextMemcpyExp(nullptr, dst_ptr, src_ptr, allocation_size));
}

TEST_P(urUSMContextMemcpyExpTestDevice, InvalidNullPtrs) {
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_POINTER,
      urUSMContextMemcpyExp(context, nullptr, src_ptr, allocation_size));
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_POINTER,
      urUSMContextMemcpyExp(context, dst_ptr, nullptr, allocation_size));
}

TEST_P(urUSMContextMemcpyExpTestDevice, InvalidZeroSize) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urUSMContextMemcpyExp(context, dst_ptr, src_ptr, 0));
}

struct urUSMContextMemcpyExpTestHost : urUSMContextMemcpyExpTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urUSMContextMemcpyExpTest::SetUp());

    ur_device_usm_access_capability_flags_t host_usm = 0;
    ASSERT_SUCCESS(uur::GetDeviceUSMHostSupport(device, host_usm));
    if (!host_usm) {
      GTEST_SKIP() << "Host USM is not supported";
    }

    ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, nullptr, allocation_size,
                                  reinterpret_cast<void **>(&src_ptr)));
    ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, nullptr, allocation_size,
                                  reinterpret_cast<void **>(&dst_ptr)));
    initAllocations();
  }
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urUSMContextMemcpyExpTestHost);

TEST_P(urUSMContextMemcpyExpTestHost, Success) {
  ASSERT_SUCCESS(
      urUSMContextMemcpyExp(context, dst_ptr, src_ptr, allocation_size));
  verifyData();
}

struct urUSMContextMemcpyExpTestShared : urUSMContextMemcpyExpTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urUSMContextMemcpyExpTest::SetUp());

    ur_device_usm_access_capability_flags_t shared_usm_single = 0;

    ASSERT_SUCCESS(
        uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_single));

    if (!shared_usm_single) {
      GTEST_SKIP() << "Shared USM is not supported by the device.";
    }

    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                    allocation_size,
                                    reinterpret_cast<void **>(&src_ptr)));
    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                    allocation_size,
                                    reinterpret_cast<void **>(&dst_ptr)));
    initAllocations();
  }
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urUSMContextMemcpyExpTestShared);

TEST_P(urUSMContextMemcpyExpTestShared, Success) {
  ASSERT_SUCCESS(
      urUSMContextMemcpyExp(context, dst_ptr, src_ptr, allocation_size));
  verifyData();
}
