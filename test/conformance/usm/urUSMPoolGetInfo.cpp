// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ur_api.h"
#include <uur/fixtures.h>

using urUSMPoolGetInfoTest = uur::urUSMPoolTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urUSMPoolGetInfoTest);

TEST_P(urUSMPoolGetInfoTest, SuccessReferenceCount) {
  size_t property_size = 0;
  ur_usm_pool_info_t property_name = UR_USM_POOL_INFO_REFERENCE_COUNT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urUSMPoolGetInfo(pool, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(uint32_t), property_size);

  uint32_t returned_reference_count = 0;
  ASSERT_SUCCESS(urUSMPoolGetInfo(pool, property_name, property_size,
                                  &returned_reference_count, nullptr));

  ASSERT_GT(returned_reference_count, 0U);
}

TEST_P(urUSMPoolGetInfoTest, SuccessContext) {
  size_t property_size = 0;
  ur_usm_pool_info_t property_name = UR_USM_POOL_INFO_CONTEXT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urUSMPoolGetInfo(pool, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(ur_context_handle_t), property_size);

  ur_context_handle_t returned_context = nullptr;
  ASSERT_SUCCESS(urUSMPoolGetInfo(pool, property_name, property_size,
                                  &returned_context, nullptr));

  ASSERT_EQ(context, returned_context);
}

TEST_P(urUSMPoolGetInfoTest, InvalidNullHandlePool) {
  ur_context_handle_t context = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urUSMPoolGetInfo(nullptr, UR_USM_POOL_INFO_CONTEXT,
                                    sizeof(ur_context_handle_t), &context,
                                    nullptr));
}

TEST_P(urUSMPoolGetInfoTest, InvalidEnumerationProperty) {
  ur_context_handle_t context = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urUSMPoolGetInfo(pool, UR_USM_POOL_INFO_FORCE_UINT32,
                                    sizeof(ur_context_handle_t), &context,
                                    nullptr));
}

TEST_P(urUSMPoolGetInfoTest, InvalidSizeZero) {
  ur_context_handle_t context = nullptr;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_SIZE,
      urUSMPoolGetInfo(pool, UR_USM_POOL_INFO_CONTEXT, 0, &context, nullptr));
}

TEST_P(urUSMPoolGetInfoTest, InvalidSizeTooSmall) {
  ur_context_handle_t context = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urUSMPoolGetInfo(pool, UR_USM_POOL_INFO_CONTEXT,
                                    sizeof(ur_context_handle_t) - 1, &context,
                                    nullptr));
}

TEST_P(urUSMPoolGetInfoTest, InvalidNullPointerPropValue) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urUSMPoolGetInfo(pool, UR_USM_POOL_INFO_CONTEXT,
                                    sizeof(ur_context_handle_t), nullptr,
                                    nullptr));
}

TEST_P(urUSMPoolGetInfoTest, InvalidNullPointerPropSizeRet) {
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_POINTER,
      urUSMPoolGetInfo(pool, UR_USM_POOL_INFO_CONTEXT, 0, nullptr, nullptr));
}
