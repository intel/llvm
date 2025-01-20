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
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urUSMPoolCreateTest);

TEST_P(urUSMPoolCreateTest, Success) {
  ur_usm_pool_desc_t pool_desc{UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr, 0};
  ur_usm_pool_handle_t pool = nullptr;
  ASSERT_SUCCESS(urUSMPoolCreate(context, &pool_desc, &pool));
  ASSERT_NE(pool, nullptr);
  EXPECT_SUCCESS(urUSMPoolRelease(pool));
}

TEST_P(urUSMPoolCreateTest, SuccessWithFlag) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{});

  ur_usm_pool_desc_t pool_desc{UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr,
                               UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK};
  ur_usm_pool_handle_t pool = nullptr;
  ASSERT_SUCCESS(urUSMPoolCreate(context, &pool_desc, &pool));
  ASSERT_NE(pool, nullptr);
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
