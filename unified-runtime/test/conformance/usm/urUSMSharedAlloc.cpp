// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "helpers.h"
#include <limits>
#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urUSMSharedAllocTest : uur::urUSMAllocTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urUSMAllocTest::SetUp());
    ur_device_usm_access_capability_flags_t shared_usm_cross = 0;
    ur_device_usm_access_capability_flags_t shared_usm_single = 0;

    ASSERT_SUCCESS(
        uur::GetDeviceUSMCrossSharedSupport(device, shared_usm_cross));
    ASSERT_SUCCESS(
        uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_single));

    if (!(shared_usm_cross || shared_usm_single)) {
      GTEST_SKIP() << "Shared USM is not supported by the device.";
    }
  }
};

// The 0 value parameters are not relevant for urUSMSharedAllocTest tests, they
// are used below in urUSMSharedAllocAlignmentTest for allocation size and
// alignment values
UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urUSMSharedAllocTest,
    testing::Combine(
        testing::ValuesIn(uur::BoolTestParam::makeBoolParam("UsePool")),
        testing::Values(0), testing::Values(0),
        testing::ValuesIn(uur::usm_advice_test_parameters)),
    uur::printUSMAllocTestString<urUSMSharedAllocTest>);

TEST_P(urUSMSharedAllocTest, Success) {
  allocation_size = sizeof(int);

  ASSERT_SUCCESS(
      urUSMSharedAlloc(context, device, nullptr, pool, allocation_size, &ptr));
  ASSERT_NE(ptr, nullptr);

  ur_event_handle_t event = nullptr;
  uint8_t pattern = 0;
  EXPECT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                  allocation_size, 0, nullptr, &event));
  EXPECT_SUCCESS(urEventWait(1, &event));

  ASSERT_SUCCESS(urUSMFree(context, ptr));
  EXPECT_SUCCESS(urEventRelease(event));
}

TEST_P(urUSMSharedAllocTest, SuccessWithDescriptors) {
  const ur_usm_device_desc_t usm_device_desc{UR_STRUCTURE_TYPE_USM_DEVICE_DESC,
                                             nullptr,
                                             /* device flags */ 0};

  const ur_usm_host_desc_t usm_host_desc{UR_STRUCTURE_TYPE_USM_HOST_DESC,
                                         &usm_device_desc,
                                         /* host flags */ 0};

  const ur_usm_desc_t usm_desc{UR_STRUCTURE_TYPE_USM_DESC, &usm_host_desc,
                               advice_flags, alignment};

  allocation_size = sizeof(int);

  ASSERT_SUCCESS(urUSMSharedAlloc(context, device, &usm_desc, pool,
                                  allocation_size, &ptr));
  ASSERT_NE(ptr, nullptr);

  ur_event_handle_t event = nullptr;
  uint8_t pattern = 0;
  EXPECT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                  allocation_size, 0, nullptr, &event));
  EXPECT_SUCCESS(urEventWait(1, &event));

  ASSERT_SUCCESS(urUSMFree(context, ptr));
  EXPECT_SUCCESS(urEventRelease(event));
}

TEST_P(urUSMSharedAllocTest, SuccessWithMultipleAdvices) {
  const ur_usm_desc_t usm_desc{
      UR_STRUCTURE_TYPE_USM_DESC, nullptr,
      /* mem advice flags */ UR_USM_ADVICE_FLAG_SET_READ_MOSTLY |
          UR_USM_ADVICE_FLAG_BIAS_CACHED,
      alignment};

  allocation_size = sizeof(int);

  ASSERT_SUCCESS(urUSMSharedAlloc(context, device, &usm_desc, pool,
                                  allocation_size, &ptr));
  ASSERT_NE(ptr, nullptr);

  ur_event_handle_t event = nullptr;
  uint8_t pattern = 0;
  EXPECT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                  allocation_size, 0, nullptr, &event));
  EXPECT_SUCCESS(urEventWait(1, &event));

  ASSERT_SUCCESS(urUSMFree(context, ptr));
  EXPECT_SUCCESS(urEventRelease(event));
}

TEST_P(urUSMSharedAllocTest, InvalidNullHandleContext) {
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_HANDLE,
      urUSMSharedAlloc(nullptr, device, nullptr, pool, sizeof(int), &ptr));
}

TEST_P(urUSMSharedAllocTest, InvalidNullHandleDevice) {
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_HANDLE,
      urUSMSharedAlloc(context, nullptr, nullptr, pool, sizeof(int), &ptr));
}

TEST_P(urUSMSharedAllocTest, InvalidNullPtrMem) {
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_POINTER,
      urUSMSharedAlloc(context, device, nullptr, pool, sizeof(int), nullptr));
}

TEST_P(urUSMSharedAllocTest, InvalidUSMSize) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{}, uur::NativeCPU{});

  void *ptr = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_USM_SIZE,
                   urUSMSharedAlloc(context, device, nullptr, pool, 0, &ptr));

  // TODO: Producing error X from case "size is greater than
  // UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE" is currently unreliable due to
  // implementation issues for UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE
  // https://github.com/oneapi-src/unified-runtime/issues/2665
}

TEST_P(urUSMSharedAllocTest, InvalidValue) {
  ur_usm_desc_t desc = {};
  desc.stype = UR_STRUCTURE_TYPE_USM_DESC;
  desc.align = 5;
  desc.pNext = nullptr;
  desc.hints = UR_USM_ADVICE_FLAG_DEFAULT;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_VALUE,
      urUSMSharedAlloc(context, device, &desc, pool, sizeof(int), &ptr));

  desc.align = std::numeric_limits<uint32_t>::max();
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_VALUE,
      urUSMSharedAlloc(context, device, &desc, pool, sizeof(int), &ptr));
}

TEST_P(urUSMSharedAllocTest, InvalidEnumeration) {
  void *ptr = nullptr;
  ur_usm_desc_t desc = {};
  desc.stype = UR_STRUCTURE_TYPE_USM_DESC;
  desc.align = 5;
  desc.pNext = nullptr;
  desc.hints = UR_USM_ADVICE_FLAG_FORCE_UINT32;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_ENUMERATION,
      urUSMSharedAlloc(context, device, &desc, pool, sizeof(int), &ptr));
}

using urUSMSharedAllocAlignmentTest = urUSMSharedAllocTest;

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urUSMSharedAllocAlignmentTest,
    testing::Combine(
        testing::ValuesIn(uur::BoolTestParam::makeBoolParam("UsePool")),
        testing::Values(4, 8, 16, 32, 64), testing::Values(8, 512, 2048),
        testing::ValuesIn(uur::usm_advice_test_parameters)),
    uur::printUSMAllocTestString<urUSMSharedAllocAlignmentTest>);

TEST_P(urUSMSharedAllocAlignmentTest, SuccessAlignedAllocations) {
  const ur_usm_device_desc_t usm_device_desc{UR_STRUCTURE_TYPE_USM_DEVICE_DESC,
                                             nullptr,
                                             /* device flags */ 0};

  const ur_usm_host_desc_t usm_host_desc{UR_STRUCTURE_TYPE_USM_HOST_DESC,
                                         &usm_device_desc,
                                         /* host flags */ 0};

  const ur_usm_desc_t usm_desc{UR_STRUCTURE_TYPE_USM_DESC, &usm_host_desc,
                               advice_flags, alignment};

  ASSERT_SUCCESS(urUSMSharedAlloc(context, device, &usm_desc, pool,
                                  allocation_size, &ptr));
  ASSERT_NE(ptr, nullptr);

  ur_event_handle_t event = nullptr;
  uint8_t pattern = 0;
  EXPECT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                  allocation_size, 0, nullptr, &event));
  EXPECT_SUCCESS(urEventWait(1, &event));

  ASSERT_SUCCESS(urUSMFree(context, ptr));
  EXPECT_SUCCESS(urEventRelease(event));
}
