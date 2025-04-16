// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/utils.h"
#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urUSMPoolCreateTest : uur::urContextTest {
  void SetUp() {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urContextTest::SetUp());
    ur_bool_t poolSupport = false;
    ASSERT_SUCCESS(uur::GetDeviceUSMPoolSupport(device, poolSupport));
    if (!poolSupport) {
      GTEST_SKIP() << "USM pools are not supported.";
    }
  }
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urUSMPoolCreateTest);

TEST_P(urUSMPoolCreateTest, Success) {
  ur_usm_pool_desc_t pool_desc{UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr, 0};
  ur_usm_pool_handle_t pool = nullptr;
  ASSERT_SUCCESS(urUSMPoolCreate(context, &pool_desc, &pool));
  ASSERT_NE(pool, nullptr);
  EXPECT_SUCCESS(urUSMPoolRelease(pool));
}

// Manages a queue and a pointer so we can check out the memory the pool
// allocates.
struct urUSMPoolCreateWithMemTest : urUSMPoolCreateTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urUSMPoolCreateTest::SetUp());
    ASSERT_SUCCESS(urQueueCreate(context, device, nullptr, &queue));
  }

  void TearDown() override {
    if (queue) {
      ASSERT_SUCCESS(urQueueRelease(queue));
    }
    if (alloc) {
      ASSERT_SUCCESS(urUSMFree(context, alloc));
    }
  }

  ur_queue_handle_t queue = nullptr;
  void *alloc = nullptr;
};

TEST_P(urUSMPoolCreateWithMemTest, SuccessWithZeroInit) {
  // We're going to test with a device allocation.
  ur_device_usm_access_capability_flags_t deviceAllocSupport = 0;
  ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, deviceAllocSupport));
  if (!deviceAllocSupport) {
    GTEST_SKIP() << "Device USM is not supported.";
  }

  ur_usm_pool_desc_t pool_desc{UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr,
                               UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK};
  constexpr size_t allocSize = 32;
  std::vector<uint8_t> hostPtr(allocSize);
  ur_usm_pool_handle_t pool = nullptr;
  ASSERT_SUCCESS(urUSMPoolCreate(context, &pool_desc, &pool));

  void *alloc = nullptr;
  ASSERT_SUCCESS(
      urUSMDeviceAlloc(context, device, nullptr, pool, allocSize, &alloc));
  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, true, hostPtr.data(), alloc,
                                    allocSize, 0, nullptr, nullptr));

  // Our allocation should be zero-initialized.
  for (auto &c : hostPtr) {
    ASSERT_EQ(c, 0);
  }
  EXPECT_SUCCESS(urUSMPoolRelease(pool));
}

TEST_P(urUSMPoolCreateTest, InvalidNullHandleContext) {
  ur_usm_pool_desc_t pool_desc{UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr,
                               UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK};
  ur_usm_pool_handle_t pool = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urUSMPoolCreate(nullptr, &pool_desc, &pool));
}

TEST_P(urUSMPoolCreateTest, InvalidNullPointerPoolDesc) {
  ur_usm_pool_handle_t pool = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urUSMPoolCreate(context, nullptr, &pool));
}

TEST_P(urUSMPoolCreateTest, InvalidNullPointerPool) {
  ur_usm_pool_desc_t pool_desc{UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr,
                               UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK};
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urUSMPoolCreate(context, &pool_desc, nullptr));
}

TEST_P(urUSMPoolCreateTest, InvalidEnumerationFlags) {
  ur_usm_pool_desc_t pool_desc{UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr,
                               UR_USM_POOL_FLAG_FORCE_UINT32};
  ur_usm_pool_handle_t pool = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urUSMPoolCreate(context, &pool_desc, &pool));
}
