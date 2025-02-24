// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
struct urL0EnqueueAllocTest : uur::urKernelExecutionTest {
  void SetUp() {
    program_name = "fill_usm";
    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
  }

  void ValidateEnqueueFree(void *ptr) {
    ur_event_handle_t freeEvent = nullptr;
    ASSERT_NE(ptr, nullptr);
    ASSERT_SUCCESS(
        urEnqueueUSMFreeExp(queue, nullptr, ptr, 0, nullptr, &freeEvent));
    ASSERT_NE(freeEvent, nullptr);
    ASSERT_SUCCESS(urQueueFinish(queue));
  }

  static constexpr size_t ARRAY_SIZE = 16;
  static constexpr uint32_t DATA = 0xC0FFEE;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urL0EnqueueAllocTest);

TEST_P(urL0EnqueueAllocTest, SuccessHost) {
  ur_device_usm_access_capability_flags_t hostUSMSupport = 0;
  ASSERT_SUCCESS(uur::GetDeviceUSMHostSupport(device, hostUSMSupport));
  if (!hostUSMSupport) {
    GTEST_SKIP() << "Host USM is not supported.";
  }

  void *ptr = nullptr;
  ur_event_handle_t allocEvent = nullptr;
  ASSERT_SUCCESS(urEnqueueUSMHostAllocExp(queue, nullptr, sizeof(uint32_t),
                                          nullptr, 0, nullptr, &ptr,
                                          &allocEvent));
  ASSERT_SUCCESS(urQueueFinish(queue));
  ASSERT_NE(ptr, nullptr);
  ASSERT_NE(allocEvent, nullptr);
  *(uint32_t *)ptr = DATA;
  ValidateEnqueueFree(ptr);
}

// Disable temporarily until user pool handling is implemented
// TEST_P(urL0EnqueueAllocTest, SuccessHostPoolAlloc) {
//     ur_device_usm_access_capability_flags_t hostUSMSupport = 0;
//     ASSERT_SUCCESS(uur::GetDeviceUSMHostSupport(device, hostUSMSupport));
//     if (!hostUSMSupport) {
//         GTEST_SKIP() << "Host USM is not supported.";
//     }

//     ur_usm_pool_handle_t pool = nullptr;
//     ASSERT_SUCCESS(urUSMPoolCreate(context, nullptr, &pool));

//     void *ptr = nullptr;
//     ur_event_handle_t allocEvent = nullptr;
//     ASSERT_SUCCESS(urEnqueueUSMHostAllocExp(queue, pool, sizeof(uint32_t),
//     nullptr,
//                                             0, nullptr, &ptr, &allocEvent));
//     ASSERT_SUCCESS(urQueueFinish(queue));
//     ASSERT_NE(ptr, nullptr);
//     ASSERT_NE(allocEvent, nullptr);
//     *static_cast<uint32_t *>(ptr) = DATA;
//     ValidateEnqueueFree(ptr, pool);
// }

TEST_P(urL0EnqueueAllocTest, SuccessDevice) {
  ur_device_usm_access_capability_flags_t deviceUSMSupport = 0;
  ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, deviceUSMSupport));
  if (!(deviceUSMSupport & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
    GTEST_SKIP() << "Device USM is not supported.";
  }

  void *ptr = nullptr;
  ur_event_handle_t allocEvent = nullptr;
  ASSERT_SUCCESS(
      urEnqueueUSMDeviceAllocExp(queue, nullptr, ARRAY_SIZE * sizeof(uint32_t),
                                 nullptr, 0, nullptr, &ptr, &allocEvent));
  ASSERT_NE(ptr, nullptr);
  ASSERT_NE(allocEvent, nullptr);
  ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, ptr));
  ASSERT_SUCCESS(urKernelSetArgValue(kernel, 1, sizeof(DATA), nullptr, &DATA));
  Launch1DRange(ARRAY_SIZE);
  ValidateEnqueueFree(ptr);
}

TEST_P(urL0EnqueueAllocTest, SuccessDeviceRepeat) {
  ur_device_usm_access_capability_flags_t deviceUSMSupport = 0;
  ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, deviceUSMSupport));
  if (!(deviceUSMSupport & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
    GTEST_SKIP() << "Device USM is not supported.";
  }

  void *ptr = nullptr;
  ASSERT_SUCCESS(
      urEnqueueUSMDeviceAllocExp(queue, nullptr, ARRAY_SIZE * sizeof(uint32_t),
                                 nullptr, 0, nullptr, &ptr, nullptr));
  ASSERT_NE(ptr, nullptr);
  ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, ptr));
  ASSERT_SUCCESS(urKernelSetArgValue(kernel, 1, sizeof(DATA), nullptr, &DATA));
  Launch1DRange(ARRAY_SIZE);
  ASSERT_SUCCESS(urEnqueueUSMFreeExp(queue, nullptr, ptr, 0, nullptr, nullptr));

  void *ptr2 = nullptr;
  ASSERT_SUCCESS(
      urEnqueueUSMDeviceAllocExp(queue, nullptr, ARRAY_SIZE * sizeof(uint32_t),
                                 nullptr, 0, nullptr, &ptr2, nullptr));
  ASSERT_NE(ptr, nullptr);
  ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, ptr2));
  ASSERT_SUCCESS(urKernelSetArgValue(kernel, 1, sizeof(DATA), nullptr, &DATA));
  Launch1DRange(ARRAY_SIZE);
  ValidateEnqueueFree(ptr2);
}

TEST_P(urL0EnqueueAllocTest, SuccessShared) {
  ur_device_usm_access_capability_flags_t sharedUSMSupport = 0;
  ASSERT_SUCCESS(
      uur::GetDeviceUSMSingleSharedSupport(device, sharedUSMSupport));
  if (!(sharedUSMSupport & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
    GTEST_SKIP() << "Shared USM is not supported.";
  }

  void *ptr = nullptr;
  ur_event_handle_t allocEvent = nullptr;
  ASSERT_SUCCESS(
      urEnqueueUSMSharedAllocExp(queue, nullptr, ARRAY_SIZE * sizeof(uint32_t),
                                 nullptr, 0, nullptr, &ptr, &allocEvent));
  ASSERT_NE(ptr, nullptr);
  ASSERT_NE(allocEvent, nullptr);
  ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, nullptr, ptr));
  ASSERT_SUCCESS(urKernelSetArgValue(kernel, 1, sizeof(DATA), nullptr, &DATA));
  Launch1DRange(ARRAY_SIZE);
  ValidateEnqueueFree(ptr);
}
