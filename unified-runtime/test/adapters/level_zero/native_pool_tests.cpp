// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include <uur/fixtures.h>

using urL0NativePoolTest = uur::urContextTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urL0NativePoolTest);

TEST_P(urL0NativePoolTest, SuccessHost) {
  ur_device_usm_access_capability_flags_t hostUSMSupport = 0;
  ASSERT_SUCCESS(uur::GetDeviceUSMHostSupport(device, hostUSMSupport));
  if (!hostUSMSupport) {
    GTEST_SKIP() << "Host USM is not supported.";
  }

  void *ptr = nullptr;
  size_t allocSize = sizeof(int) * 1024;
  ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, nullptr, allocSize, &ptr));
  ASSERT_NE(ptr, nullptr);

  // Set native pool descriptor buffer to the USM allocation
  ur_usm_pool_native_desc_t nativePoolDesc{};
  nativePoolDesc.stype = UR_STRUCTURE_TYPE_USM_POOL_NATIVE_DESC;
  nativePoolDesc.pNext = nullptr;
  nativePoolDesc.pMem = ptr;
  nativePoolDesc.size = allocSize;
  nativePoolDesc.memType = UR_USM_TYPE_HOST;
  nativePoolDesc.device = nullptr;

  ur_usm_pool_desc_t poolDesc{};
  poolDesc.stype = UR_STRUCTURE_TYPE_USM_POOL_DESC;
  poolDesc.pNext = &nativePoolDesc;
  poolDesc.flags = 0;

  ur_usm_pool_handle_t pool = nullptr;
  ASSERT_SUCCESS(urUSMPoolCreate(context, &poolDesc, &pool));

  void *samePtr = nullptr;
  ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, pool, allocSize, &samePtr));
  ASSERT_EQ(ptr, samePtr);
  ASSERT_SUCCESS(urUSMFree(context, samePtr));

  ASSERT_SUCCESS(urUSMPoolRelease(pool));
  ASSERT_SUCCESS(urUSMFree(context, ptr));
}

// TEST_P(urL0NativePoolTest, FailSize) {
//   ur_device_usm_access_capability_flags_t hostUSMSupport = 0;
//   ASSERT_SUCCESS(uur::GetDeviceUSMHostSupport(device, hostUSMSupport));
//   if (!hostUSMSupport) {
//     GTEST_SKIP() << "Host USM is not supported.";
//   }

//   void *ptr = nullptr;
//   size_t allocSize = sizeof(int) * 1024;
//   ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, nullptr, allocSize, &ptr));
//   ASSERT_NE(ptr, nullptr);

//   // Set native pool descriptor buffer to the USM allocation
//   ur_usm_pool_native_desc_t nativePoolDesc{};
//   nativePoolDesc.stype = UR_STRUCTURE_TYPE_USM_POOL_NATIVE_DESC;
//   nativePoolDesc.pNext = nullptr;
//   nativePoolDesc.pMem = ptr;
//   nativePoolDesc.size = allocSize;
//   nativePoolDesc.memType = UR_USM_TYPE_HOST;
//   nativePoolDesc.device = nullptr;

//   ur_usm_pool_desc_t poolDesc{};
//   poolDesc.stype = UR_STRUCTURE_TYPE_USM_POOL_DESC;
//   poolDesc.pNext = &nativePoolDesc;
//   poolDesc.flags = 0;

//   ur_usm_pool_handle_t pool = nullptr;
//   ASSERT_SUCCESS(urUSMPoolCreate(context, &poolDesc, &pool));

//   void *samePtr = nullptr;
//   ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, pool, allocSize, &samePtr));
//   ASSERT_EQ(ptr, samePtr);
//   ASSERT_SUCCESS(urUSMFree(context, samePtr));

//   ASSERT_SUCCESS(urUSMPoolRelease(pool));
//   ASSERT_SUCCESS(urUSMFree(context, ptr));
// }
