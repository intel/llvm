// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/optional_queries.h>

using urMemImageGetInfoTest = uur::urMemImageTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urMemImageGetInfoTest);

bool operator==(ur_image_format_t lhs, ur_image_format_t rhs) {
  return lhs.channelOrder == rhs.channelOrder &&
         lhs.channelType == rhs.channelType;
}

TEST_P(urMemImageGetInfoTest, SuccessFormat) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  size_t property_size = 0;
  ur_image_info_t property_name = UR_IMAGE_INFO_FORMAT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urMemImageGetInfo(image, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(ur_image_format_t), property_size);

  ur_image_format_t property_value = {UR_IMAGE_CHANNEL_ORDER_FORCE_UINT32,
                                      UR_IMAGE_CHANNEL_TYPE_FORCE_UINT32};

  ASSERT_SUCCESS(urMemImageGetInfo(image, property_name, property_size,
                                   &property_value, nullptr));

  ASSERT_TRUE(property_value == image_format);
}

TEST_P(urMemImageGetInfoTest, SuccessElementSize) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  size_t property_size = 0;
  ur_image_info_t property_name = UR_IMAGE_INFO_ELEMENT_SIZE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urMemImageGetInfo(image, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(size_t), property_size);

  size_t property_value = 999;
  ASSERT_SUCCESS(urMemImageGetInfo(image, property_name, property_size,
                                   &property_value, nullptr));

  ASSERT_NE(property_value, 999);
}

TEST_P(urMemImageGetInfoTest, SuccessRowPitch) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  size_t property_size = 0;
  ur_image_info_t property_name = UR_IMAGE_INFO_ROW_PITCH;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urMemImageGetInfo(image, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(size_t), property_size);

  size_t property_value = 999;
  ASSERT_SUCCESS(urMemImageGetInfo(image, property_name, property_size,
                                   &property_value, nullptr));

  ASSERT_TRUE(property_value == image_desc.rowPitch ||
              property_value == (4 * sizeof(uint8_t)) * image_desc.width);
}

TEST_P(urMemImageGetInfoTest, SuccessSlicePitch) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  size_t property_size = 0;
  ur_image_info_t property_name = UR_IMAGE_INFO_SLICE_PITCH;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urMemImageGetInfo(image, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(size_t), property_size);

  size_t property_value = 999;
  ASSERT_SUCCESS(urMemImageGetInfo(image, property_name, property_size,
                                   &property_value, nullptr));

  ASSERT_EQ(property_value, image_desc.slicePitch);
}

TEST_P(urMemImageGetInfoTest, SuccessWidth) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  size_t property_size = 0;
  ur_image_info_t property_name = UR_IMAGE_INFO_WIDTH;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urMemImageGetInfo(image, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(size_t), property_size);

  size_t property_value = 999;
  ASSERT_SUCCESS(urMemImageGetInfo(image, property_name, property_size,
                                   &property_value, nullptr));

  ASSERT_EQ(property_value, image_desc.width);
}

TEST_P(urMemImageGetInfoTest, SuccessHeight) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  size_t property_size = 0;
  ur_image_info_t property_name = UR_IMAGE_INFO_HEIGHT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urMemImageGetInfo(image, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(size_t), property_size);

  size_t property_value = 999;
  ASSERT_SUCCESS(urMemImageGetInfo(image, property_name, property_size,
                                   &property_value, nullptr));

  ASSERT_EQ(property_value, image_desc.height);
}

TEST_P(urMemImageGetInfoTest, SuccessDepth) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  size_t property_size = 0;
  ur_image_info_t property_name = UR_IMAGE_INFO_DEPTH;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urMemImageGetInfo(image, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(size_t), property_size);

  size_t property_value = 999;
  ASSERT_SUCCESS(urMemImageGetInfo(image, property_name, property_size,
                                   &property_value, nullptr));

  ASSERT_TRUE(property_value == image_desc.depth || property_value == 0);
}

TEST_P(urMemImageGetInfoTest, SuccessArraySize) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  size_t property_size = 0;
  ur_image_info_t property_name = UR_IMAGE_INFO_ARRAY_SIZE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urMemImageGetInfo(image, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(size_t), property_size);

  size_t property_value = 999;
  ASSERT_SUCCESS(urMemImageGetInfo(image, property_name, property_size,
                                   &property_value, nullptr));

  ASSERT_TRUE(property_value == image_desc.depth || property_value == 0);
}

TEST_P(urMemImageGetInfoTest, SuccessNumMipMaps) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  size_t property_size = 0;
  ur_image_info_t property_name = UR_IMAGE_INFO_NUM_MIP_LEVELS;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urMemImageGetInfo(image, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(uint32_t), property_size);

  uint32_t property_value = 999;
  ASSERT_SUCCESS(urMemImageGetInfo(image, property_name, property_size,
                                   &property_value, nullptr));

  ASSERT_EQ(property_value, image_desc.numMipLevel);
}

TEST_P(urMemImageGetInfoTest, SuccessNumSamples) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  size_t property_size = 0;
  ur_image_info_t property_name = UR_IMAGE_INFO_NUM_SAMPLES;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urMemImageGetInfo(image, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(uint32_t), property_size);

  uint32_t property_value = 999;
  ASSERT_SUCCESS(urMemImageGetInfo(image, property_name, property_size,
                                   &property_value, nullptr));

  ASSERT_EQ(property_value, image_desc.numSamples);
}

TEST_P(urMemImageGetInfoTest, InvalidNullHandleImage) {
  size_t property_size = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urMemImageGetInfo(nullptr, UR_IMAGE_INFO_FORMAT,
                                     sizeof(size_t), &property_size, nullptr));
}

TEST_P(urMemImageGetInfoTest, InvalidEnumerationImageInfoType) {
  size_t property_size = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urMemImageGetInfo(image, UR_IMAGE_INFO_FORCE_UINT32,
                                     sizeof(size_t), &property_size, nullptr));
}

TEST_P(urMemImageGetInfoTest, InvalidSizeZero) {
  size_t property_size = 0;
  ASSERT_EQ_RESULT(urMemImageGetInfo(image, UR_IMAGE_INFO_FORMAT, 0,
                                     &property_size, nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urMemImageGetInfoTest, InvalidSizeSmall) {
  // This fail is specific to the "Multi device testing" ci job.
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  int property_size = 0;
  ASSERT_EQ_RESULT(urMemImageGetInfo(image, UR_IMAGE_INFO_FORMAT,
                                     sizeof(property_size) - 1, &property_size,
                                     nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urMemImageGetInfoTest, InvalidNullPointerParamValue) {
  size_t property_size = 0;
  ASSERT_EQ_RESULT(urMemImageGetInfo(image, UR_IMAGE_INFO_FORMAT,
                                     sizeof(property_size), nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urMemImageGetInfoTest, InvalidNullPointerPropSizeRet) {
  ASSERT_EQ_RESULT(
      urMemImageGetInfo(image, UR_IMAGE_INFO_FORMAT, 0, nullptr, nullptr),
      UR_RESULT_ERROR_INVALID_NULL_POINTER);
}
