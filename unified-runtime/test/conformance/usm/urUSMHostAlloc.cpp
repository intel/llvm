// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "helpers.h"
#include <cstring>
#include <limits>
#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urUSMHostAllocTest : uur::urUSMAllocTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urUSMAllocTest::SetUp());
    ASSERT_SUCCESS(uur::GetDeviceUSMHostSupport(device, USMSupport));
    if (!USMSupport) {
      GTEST_SKIP() << "Host USM is not supported.";
    }
  }
};

// The 0 value parameters are not relevant for urUSMHostAllocTest tests, they
// are used below in urUSMHostAllocAlignmentTest for allocation size and
// alignment values
UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urUSMHostAllocTest,
    testing::Combine(
        testing::ValuesIn(uur::BoolTestParam::makeBoolParam("UsePool")),
        testing::Values(0), testing::Values(0),
        ::testing::ValuesIn(uur::usm_advice_test_parameters)),
    uur::printUSMAllocTestString<urUSMHostAllocTest>);

TEST_P(urUSMHostAllocTest, Success) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  allocation_size = sizeof(int);
  ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, pool, sizeof(int),
                                reinterpret_cast<void **>(&ptr)));
  ASSERT_NE(ptr, nullptr);

  // Set 0
  ur_event_handle_t event = nullptr;

  uint8_t pattern = 0;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                  allocation_size, 0, nullptr, &event));
  EXPECT_SUCCESS(urQueueFlush(queue));
  ASSERT_SUCCESS(urEventWait(1, &event));
  EXPECT_SUCCESS(urEventRelease(event));
  ASSERT_EQ(*reinterpret_cast<int *>(ptr), 0);

  // Set 1, in all bytes of int
  pattern = 1;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                  allocation_size, 0, nullptr, &event));
  EXPECT_SUCCESS(urQueueFlush(queue));
  ASSERT_SUCCESS(urEventWait(1, &event));
  EXPECT_SUCCESS(urEventRelease(event));

  // replicate it on host
  int set_data = 0;
  std::memset(&set_data, 1, allocation_size);
  ASSERT_EQ(*reinterpret_cast<int *>(ptr), set_data);

  ASSERT_SUCCESS(urUSMFree(context, ptr));
}

TEST_P(urUSMHostAllocTest, SuccessWithDescriptors) {
  const ur_usm_host_desc_t usm_host_desc{UR_STRUCTURE_TYPE_USM_HOST_DESC,
                                         nullptr,
                                         /* host flags */ 0};

  const ur_usm_desc_t usm_desc{UR_STRUCTURE_TYPE_USM_DESC, &usm_host_desc,
                               advice_flags, alignment};

  allocation_size = sizeof(int);

  ASSERT_SUCCESS(
      urUSMHostAlloc(context, &usm_desc, pool, allocation_size, &ptr));
  ASSERT_NE(ptr, nullptr);

  ur_event_handle_t event = nullptr;
  uint8_t pattern = 0;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                  allocation_size, 0, nullptr, &event));
  ASSERT_SUCCESS(urEventWait(1, &event));

  ASSERT_SUCCESS(urUSMFree(context, ptr));
  EXPECT_SUCCESS(urEventRelease(event));
}

TEST_P(urUSMHostAllocTest, InvalidNullHandleContext) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urUSMHostAlloc(nullptr, nullptr, pool, sizeof(int), &ptr));
}

TEST_P(urUSMHostAllocTest, InvalidNullPtrMem) {
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_POINTER,
      urUSMHostAlloc(context, nullptr, pool, sizeof(int), nullptr));
}

TEST_P(urUSMHostAllocTest, InvalidUSMSize) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{}, uur::NativeCPU{});

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_USM_SIZE,
                   urUSMHostAlloc(context, nullptr, pool, 0, &ptr));

  // TODO: Producing error X from case "size is greater than
  // UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE" is currently unreliable due to
  // implementation issues for UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE
  // https://github.com/oneapi-src/unified-runtime/issues/2665
}

TEST_P(urUSMHostAllocTest, InvalidValue) {
  ur_usm_desc_t desc = {};
  desc.stype = UR_STRUCTURE_TYPE_USM_DESC;
  desc.align = 5;
  desc.hints = UR_USM_ADVICE_FLAG_DEFAULT;
  desc.pNext = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_VALUE,
                   urUSMHostAlloc(context, &desc, pool, sizeof(int), &ptr));

  desc.align = std::numeric_limits<uint32_t>::max();
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_VALUE,
                   urUSMHostAlloc(context, &desc, pool, sizeof(int), &ptr));
}

TEST_P(urUSMHostAllocTest, InvalidEnumeration) {
  ur_usm_desc_t desc = {};
  desc.stype = UR_STRUCTURE_TYPE_USM_DESC;
  desc.align = 5;
  desc.hints = UR_USM_ADVICE_FLAG_FORCE_UINT32;
  desc.pNext = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urUSMHostAlloc(context, &desc, pool, sizeof(int), &ptr));
}

using urUSMHostAllocAlignmentTest = urUSMHostAllocTest;

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urUSMHostAllocAlignmentTest,
    testing::Combine(
        testing::ValuesIn(uur::BoolTestParam::makeBoolParam("UsePool")),
        testing::Values(4, 8, 16, 32, 64), testing::Values(8, 512, 2048),
        ::testing::ValuesIn(uur::usm_advice_test_parameters)),
    uur::printUSMAllocTestString<urUSMHostAllocAlignmentTest>);

TEST_P(urUSMHostAllocAlignmentTest, SuccessAlignedAllocations) {
  const ur_usm_host_desc_t usm_host_desc{UR_STRUCTURE_TYPE_USM_HOST_DESC,
                                         nullptr,
                                         /* host flags */ 0};

  const ur_usm_desc_t usm_desc{UR_STRUCTURE_TYPE_USM_DESC, &usm_host_desc,
                               advice_flags, alignment};

  ASSERT_SUCCESS(
      urUSMHostAlloc(context, &usm_desc, pool, allocation_size, &ptr));
  ASSERT_NE(ptr, nullptr);

  ur_event_handle_t event = nullptr;
  uint8_t pattern = 0;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                  allocation_size, 0, nullptr, &event));
  ASSERT_SUCCESS(urEventWait(1, &event));

  ASSERT_SUCCESS(urUSMFree(context, ptr));
  EXPECT_SUCCESS(urEventRelease(event));
}
