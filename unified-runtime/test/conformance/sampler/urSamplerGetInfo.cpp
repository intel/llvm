// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/known_failure.h>

using urSamplerGetInfoTest = uur::urSamplerTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urSamplerGetInfoTest);

TEST_P(urSamplerGetInfoTest, SuccessReferenceCount) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  size_t property_size = 0;
  const ur_sampler_info_t property_name = UR_SAMPLER_INFO_REFERENCE_COUNT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urSamplerGetInfo(sampler, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(uint32_t), property_size);

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urSamplerGetInfo(sampler, property_name,
                                              property_size, &property_value,
                                              nullptr),
                             property_value);

  ASSERT_GT(property_value, 0U);
}

TEST_P(urSamplerGetInfoTest, SuccessContext) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  size_t property_size = 0;
  const ur_sampler_info_t property_name = UR_SAMPLER_INFO_CONTEXT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urSamplerGetInfo(sampler, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(ur_context_handle_t), property_size);

  ur_context_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urSamplerGetInfo(sampler, property_name, property_size,
                                  &property_value, nullptr));

  ASSERT_EQ(property_value, context);
}

TEST_P(urSamplerGetInfoTest, SuccessRoundtripContext) {
  const ur_sampler_info_t property_name = UR_SAMPLER_INFO_CONTEXT;
  size_t property_size = sizeof(ur_context_handle_t);

  ur_native_handle_t native_sampler;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urSamplerGetNativeHandle(sampler, &native_sampler));

  ur_sampler_handle_t from_native_sampler;
  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(urSamplerCreateWithNativeHandle(
      native_sampler, context, nullptr, &from_native_sampler));

  ur_context_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urSamplerGetInfo(from_native_sampler, property_name,
                                  property_size, &property_value, nullptr));

  ASSERT_EQ(property_value, context);
}

TEST_P(urSamplerGetInfoTest, SuccessNormalizedCoords) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  size_t property_size = 0;
  const ur_sampler_info_t property_name = UR_SAMPLER_INFO_NORMALIZED_COORDS;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urSamplerGetInfo(sampler, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(ur_bool_t), property_size);
}

TEST_P(urSamplerGetInfoTest, SuccessAddressingMode) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  size_t property_size = 0;
  const ur_sampler_info_t property_name = UR_SAMPLER_INFO_ADDRESSING_MODE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urSamplerGetInfo(sampler, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(ur_sampler_addressing_mode_t), property_size);

  ur_sampler_addressing_mode_t property_value =
      UR_SAMPLER_ADDRESSING_MODE_FORCE_UINT32;
  ASSERT_SUCCESS(urSamplerGetInfo(sampler, property_name, property_size,
                                  &property_value, nullptr));

  ASSERT_GE(property_value, UR_SAMPLER_ADDRESSING_MODE_NONE);
  ASSERT_LT(property_value, UR_SAMPLER_ADDRESSING_MODE_FORCE_UINT32);
}

TEST_P(urSamplerGetInfoTest, SuccessFilterMode) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  size_t property_size = 0;
  const ur_sampler_info_t property_name = UR_SAMPLER_INFO_FILTER_MODE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urSamplerGetInfo(sampler, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(ur_sampler_filter_mode_t), property_size);

  ur_sampler_filter_mode_t property_value = UR_SAMPLER_FILTER_MODE_FORCE_UINT32;
  ASSERT_SUCCESS(urSamplerGetInfo(sampler, property_name, property_size,
                                  &property_value, nullptr));

  ASSERT_GE(property_value, UR_SAMPLER_FILTER_MODE_NEAREST);
  ASSERT_LT(property_value, UR_SAMPLER_FILTER_MODE_FORCE_UINT32);
}

TEST_P(urSamplerGetInfoTest, InvalidNullHandleSampler) {
  uint32_t property_value = 0;
  ASSERT_EQ_RESULT(urSamplerGetInfo(nullptr, UR_SAMPLER_INFO_REFERENCE_COUNT,
                                    sizeof(property_value), &property_value,
                                    nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urSamplerGetInfoTest, InvalidEnumerationInfo) {
  size_t property_size = 0;
  ASSERT_EQ_RESULT(urSamplerGetInfo(sampler, UR_SAMPLER_INFO_FORCE_UINT32, 0,
                                    nullptr, &property_size),
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
  ur_sampler_addressing_mode_t property_value = UR_SAMPLER_ADDRESSING_MODE_NONE;
  ASSERT_EQ_RESULT(urSamplerGetInfo(sampler, UR_SAMPLER_INFO_ADDRESSING_MODE, 0,
                                    &property_value, nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urSamplerGetInfoTest, InvalidSizePropSizeSmall) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});
  ur_sampler_addressing_mode_t property_value = UR_SAMPLER_ADDRESSING_MODE_NONE;

  ASSERT_EQ_RESULT(urSamplerGetInfo(sampler, UR_SAMPLER_INFO_ADDRESSING_MODE,
                                    sizeof(property_value) - 1, &property_value,
                                    nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);
}
