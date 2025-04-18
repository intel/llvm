// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "helpers.h"
#include "uur/utils.h"
#include <limits>
#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urUSMDeviceAllocTest : uur::urUSMAllocTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urUSMAllocTest::SetUp());
    ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, USMSupport));
    if (!USMSupport) {
      GTEST_SKIP() << "Device USM is not supported.";
    }
  }
};

// The 0 value parameters are not relevant for urUSMDeviceAllocTest tests, they
// are used below in urUSMDeviceAllocAlignmentTest for allocation size and
// alignment values
UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urUSMDeviceAllocTest,
    testing::Combine(
        testing::ValuesIn(uur::BoolTestParam::makeBoolParam("UsePool")),
        testing::Values(0), testing::Values(0),
        ::testing::ValuesIn(uur::usm_advice_test_parameters)),
    uur::printUSMAllocTestString<urUSMDeviceAllocTest>);

TEST_P(urUSMDeviceAllocTest, Success) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  allocation_size = sizeof(int);
  ASSERT_SUCCESS(
      urUSMDeviceAlloc(context, device, nullptr, pool, allocation_size, &ptr));
  ASSERT_NE(ptr, nullptr);

  ur_event_handle_t event = nullptr;
  uint8_t pattern = 0;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                  allocation_size, 0, nullptr, &event));
  EXPECT_SUCCESS(urQueueFlush(queue));
  ASSERT_SUCCESS(urEventWait(1, &event));

  ASSERT_SUCCESS(urUSMFree(context, ptr));
  EXPECT_SUCCESS(urEventRelease(event));
}

TEST_P(urUSMDeviceAllocTest, SuccessWithDescriptors) {
  const ur_usm_device_desc_t usm_device_desc{UR_STRUCTURE_TYPE_USM_DEVICE_DESC,
                                             nullptr,
                                             /* device flags */ 0};

  const ur_usm_desc_t usm_desc{UR_STRUCTURE_TYPE_USM_DESC, &usm_device_desc,
                               advice_flags, alignment};

  allocation_size = sizeof(int);

  ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, &usm_desc, pool,
                                  allocation_size, &ptr));
  ASSERT_NE(ptr, nullptr);

  ur_event_handle_t event = nullptr;
  uint8_t pattern = 0;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                  allocation_size, 0, nullptr, &event));
  ASSERT_SUCCESS(urEventWait(1, &event));

  ASSERT_SUCCESS(urUSMFree(context, ptr));
  EXPECT_SUCCESS(urEventRelease(event));
}

TEST_P(urUSMDeviceAllocTest, InvalidNullHandleContext) {
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_HANDLE,
      urUSMDeviceAlloc(nullptr, device, nullptr, pool, sizeof(int), &ptr));
}

TEST_P(urUSMDeviceAllocTest, InvalidNullHandleDevice) {
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_HANDLE,
      urUSMDeviceAlloc(context, nullptr, nullptr, pool, sizeof(int), &ptr));
}

TEST_P(urUSMDeviceAllocTest, InvalidNullPtrResult) {
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_POINTER,
      urUSMDeviceAlloc(context, device, nullptr, pool, sizeof(int), nullptr));
}

TEST_P(urUSMDeviceAllocTest, InvalidUSMSize) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{}, uur::LevelZero{},
                       uur::NativeCPU{});

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_USM_SIZE,
                   urUSMDeviceAlloc(context, device, nullptr, pool, 0, &ptr));

  // TODO: Producing error X from case "size is greater than
  // UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE" is currently unreliable due to
  // implementation issues for UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE
  // https://github.com/oneapi-src/unified-runtime/issues/2665
}

TEST_P(urUSMDeviceAllocTest, InvalidValue) {
  ur_usm_desc_t desc = {};
  desc.stype = UR_STRUCTURE_TYPE_USM_DESC;
  desc.align = 5;
  desc.pNext = nullptr;
  desc.hints = UR_USM_ADVICE_FLAG_DEFAULT;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_VALUE,
      urUSMDeviceAlloc(context, device, &desc, pool, sizeof(int), &ptr));

  desc.align = std::numeric_limits<uint32_t>::max();
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_VALUE,
      urUSMDeviceAlloc(context, device, &desc, pool, sizeof(int), &ptr));
}

TEST_P(urUSMDeviceAllocTest, InvalidEnumeration) {
  ur_usm_desc_t desc = {};
  desc.stype = UR_STRUCTURE_TYPE_USM_DESC;
  desc.align = 5;
  desc.pNext = nullptr;
  desc.hints = UR_USM_ADVICE_FLAG_FORCE_UINT32;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_ENUMERATION,
      urUSMDeviceAlloc(context, device, &desc, pool, sizeof(int), &ptr));
}

using urUSMDeviceAllocAlignmentTest = urUSMDeviceAllocTest;

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urUSMDeviceAllocAlignmentTest,
    testing::Combine(
        testing::ValuesIn(uur::BoolTestParam::makeBoolParam("UsePool")),
        testing::Values(4, 8, 16, 32, 64), testing::Values(8, 512, 2048),
        testing::ValuesIn(uur::usm_advice_test_parameters)),
    uur::printUSMAllocTestString<urUSMDeviceAllocAlignmentTest>);

TEST_P(urUSMDeviceAllocAlignmentTest, SuccessAlignedAllocations) {
  const ur_usm_device_desc_t usm_device_desc{UR_STRUCTURE_TYPE_USM_DEVICE_DESC,
                                             nullptr,
                                             /* device flags */ 0};

  const ur_usm_desc_t usm_desc{UR_STRUCTURE_TYPE_USM_DESC, &usm_device_desc,
                               advice_flags, alignment};

  ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, &usm_desc, pool,
                                  allocation_size, &ptr));
  ASSERT_NE(ptr, nullptr);

  ur_event_handle_t event = nullptr;
  uint8_t pattern = 0;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                  allocation_size, 0, nullptr, &event));
  ASSERT_SUCCESS(urEventWait(1, &event));

  ASSERT_SUCCESS(urUSMFree(context, ptr));
  EXPECT_SUCCESS(urEventRelease(event));
}
