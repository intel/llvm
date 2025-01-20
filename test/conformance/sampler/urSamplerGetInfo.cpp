// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/known_failure.h>

using urSamplerGetInfoTest = uur::urSamplerTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urSamplerGetInfoTest);

TEST_P(urSamplerGetInfoTest, SuccessReferenceCount) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  size_t property_size = 0;
  ur_sampler_info_t property_name = UR_SAMPLER_INFO_REFERENCE_COUNT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urSamplerGetInfo(sampler, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(uint32_t), property_size);

  uint32_t returned_reference_count = 0;
  ASSERT_SUCCESS(urSamplerGetInfo(sampler, property_name, property_size,
                                  &returned_reference_count, nullptr));

  ASSERT_GT(returned_reference_count, 0U);
}

TEST_P(urSamplerGetInfoTest, SuccessContext) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  size_t property_size = 0;
  ur_sampler_info_t property_name = UR_SAMPLER_INFO_CONTEXT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urSamplerGetInfo(sampler, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(ur_context_handle_t), property_size);

  ur_context_handle_t returned_context = nullptr;
  ASSERT_SUCCESS(urSamplerGetInfo(sampler, property_name, property_size,
                                  &returned_context, nullptr));

  ASSERT_EQ(returned_context, context);
}

TEST_P(urSamplerGetInfoTest, SuccessNormalizedCoords) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  size_t property_size = 0;
  ur_sampler_info_t property_name = UR_SAMPLER_INFO_NORMALIZED_COORDS;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urSamplerGetInfo(sampler, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(ur_bool_t), property_size);
}

TEST_P(urSamplerGetInfoTest, SuccessAddressingMode) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  size_t property_size = 0;
  ur_sampler_info_t property_name = UR_SAMPLER_INFO_ADDRESSING_MODE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urSamplerGetInfo(sampler, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(ur_sampler_addressing_mode_t), property_size);

  ur_sampler_addressing_mode_t returned_mode =
      UR_SAMPLER_ADDRESSING_MODE_FORCE_UINT32;
  ASSERT_SUCCESS(urSamplerGetInfo(sampler, property_name, property_size,
                                  &returned_mode, nullptr));

  ASSERT_GE(returned_mode, UR_SAMPLER_ADDRESSING_MODE_NONE);
  ASSERT_LT(returned_mode, UR_SAMPLER_ADDRESSING_MODE_FORCE_UINT32);
}

TEST_P(urSamplerGetInfoTest, SuccessFilterMode) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  size_t property_size = 0;
  ur_sampler_info_t property_name = UR_SAMPLER_INFO_FILTER_MODE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urSamplerGetInfo(sampler, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(ur_sampler_filter_mode_t), property_size);

  ur_sampler_filter_mode_t returned_mode = UR_SAMPLER_FILTER_MODE_FORCE_UINT32;
  ASSERT_SUCCESS(urSamplerGetInfo(sampler, property_name, property_size,
                                  &returned_mode, nullptr));

  ASSERT_GE(returned_mode, UR_SAMPLER_FILTER_MODE_NEAREST);
  ASSERT_LT(returned_mode, UR_SAMPLER_FILTER_MODE_FORCE_UINT32);
}

TEST_P(urSamplerGetInfoTest, InvalidNullHandleSampler) {
  uint32_t refcount = 0;
  ASSERT_EQ_RESULT(urSamplerGetInfo(nullptr, UR_SAMPLER_INFO_REFERENCE_COUNT,
                                    sizeof(refcount), &refcount, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urSamplerGetInfoTest, InvalidEnumerationInfo) {
  size_t size = 0;
  ASSERT_EQ_RESULT(urSamplerGetInfo(sampler, UR_SAMPLER_INFO_FORCE_UINT32, 0,
                                    nullptr, &size),
                   UR_RESULT_ERROR_INVALID_ENUMERATION);
}

TEST_P(urSamplerGetInfoTest, InvalidNullPointerPropSizeRet) {
  ASSERT_EQ_RESULT(urSamplerGetInfo(sampler, UR_SAMPLER_INFO_ADDRESSING_MODE, 0,
                                    nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urSamplerGetInfoTest, InvalidNullPointerPropValue) {
  ASSERT_EQ_RESULT(urSamplerGetInfo(sampler, UR_SAMPLER_INFO_ADDRESSING_MODE,
                                    sizeof(ur_sampler_addressing_mode_t),
                                    nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urSamplerGetInfoTest, InvalidSizePropSizeZero) {
  ur_sampler_addressing_mode_t mode = UR_SAMPLER_ADDRESSING_MODE_NONE;
  ASSERT_EQ_RESULT(urSamplerGetInfo(sampler, UR_SAMPLER_INFO_ADDRESSING_MODE, 0,
                                    &mode, nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urSamplerGetInfoTest, InvalidSizePropSizeSmall) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});
  ur_sampler_addressing_mode_t mode = UR_SAMPLER_ADDRESSING_MODE_NONE;

  ASSERT_EQ_RESULT(urSamplerGetInfo(sampler, UR_SAMPLER_INFO_ADDRESSING_MODE,
                                    sizeof(mode) - 1, &mode, nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);
}
