// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <unordered_map>

#include <uur/fixtures.h>
#include <uur/known_failure.h>

using urDeviceGetInfoTest = uur::urDeviceTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urDeviceGetInfoTest);

TEST_P(urDeviceGetInfoTest, SuccessDeviceType) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_TYPE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_device_type_t));

  ur_device_type_t property_value = UR_DEVICE_TYPE_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_GE(property_value, UR_DEVICE_TYPE_DEFAULT);
  ASSERT_LT(property_value, UR_DEVICE_TYPE_FORCE_UINT32);
}

TEST_P(urDeviceGetInfoTest, SuccessVendorId) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_VENDOR_ID;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessDeviceId) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_DEVICE_ID;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMaxComputeUnits) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MAX_COMPUTE_UNITS;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMaxWorkItemDimensions) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMaxWorkItemSizes) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES;

  ASSERT_SUCCESS(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_GT(property_size, 0);

  size_t dimension_property_size = 0;
  ASSERT_SUCCESS(urDeviceGetInfo(device,
                                 UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS, 0,
                                 nullptr, &dimension_property_size));
  ASSERT_EQ(dimension_property_size, sizeof(uint32_t));

  size_t max_dimensions = 0;
  ASSERT_SUCCESS(
      urDeviceGetInfo(device, UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS,
                      dimension_property_size, &max_dimensions, nullptr));
  ASSERT_GT(max_dimensions, 0);

  std::vector<size_t> max_work_item_sizes(max_dimensions);
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name,
                                 max_dimensions * sizeof(size_t),
                                 max_work_item_sizes.data(), nullptr));

  EXPECT_EQ(property_size, max_dimensions * sizeof(size_t));
}

TEST_P(urDeviceGetInfoTest, SuccessMaxWorkGroupSize) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(size_t));

  size_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessSingleFPConfig) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_SINGLE_FP_CONFIG;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_device_fp_capability_flags_t));

  ur_device_fp_capability_flags_t property_value =
      UR_DEVICE_FP_CAPABILITY_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value & UR_DEVICE_FP_CAPABILITY_FLAGS_MASK, 0);
}

TEST_P(urDeviceGetInfoTest, SuccessHalfFPConfig) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_HALF_FP_CONFIG;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_device_fp_capability_flags_t));

  ur_device_fp_capability_flags_t property_value =
      UR_DEVICE_FP_CAPABILITY_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value & UR_DEVICE_FP_CAPABILITY_FLAGS_MASK, 0);
}

TEST_P(urDeviceGetInfoTest, SuccessDoubleFPConfig) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_DOUBLE_FP_CONFIG;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_device_fp_capability_flags_t));

  ur_device_fp_capability_flags_t property_value =
      UR_DEVICE_FP_CAPABILITY_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value & UR_DEVICE_FP_CAPABILITY_FLAGS_MASK, 0);
}

TEST_P(urDeviceGetInfoTest, SuccessQueueProperties) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_QUEUE_PROPERTIES;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_queue_flags_t));

  ur_queue_flags_t property_value = UR_QUEUE_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value & UR_QUEUE_FLAGS_MASK, 0);
}

TEST_P(urDeviceGetInfoTest, SuccessPreferredVectorWidthChar) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessPreferredVectorWidthShort) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessPreferredVectorWidthInt) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessPreferredVectorWidthLong) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessPreferredVectorWidthFloat) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessPreferredVectorWidthDouble) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessPreferredVectorWidthHalf) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessNativeVectorWidthChar) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessNativeVectorWidthShort) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessNativeVectorWidthInt) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessNativeVectorWidthLong) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessNativeVectorWidthFloat) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessNativeVectorWidthDouble) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessNativeVectorWidthHalf) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMaxClockFrequency) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MAX_CLOCK_FREQUENCY;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMemoryClockRate) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MEMORY_CLOCK_RATE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessAddressBits) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_ADDRESS_BITS;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMaxMemAllocSize) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint64_t));

  uint64_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessImageSupported) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_IMAGE_SUPPORT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessMaxReadImageArgs) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MAX_READ_IMAGE_ARGS;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMaxWriteImageArgs) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMaxReadWriteImageArgs) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_MAX_READ_WRITE_IMAGE_ARGS;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessImage2DMaxWidth) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(size_t));

  size_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessImage2DMaxHeight) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(size_t));

  size_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessImage3DMaxWidth) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(size_t));

  size_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessImage3DMaxHeight) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(size_t));

  size_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessImage3DMaxDepth) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(size_t));

  size_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessImageMaxBufferSize) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(size_t));

  size_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessImageMaxArraySize) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(size_t));

  size_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMaxSamplers) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MAX_SAMPLERS;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMaxParameterSize) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MAX_PARAMETER_SIZE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(size_t));

  size_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMemoryBaseAddressAlignment) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MEM_BASE_ADDR_ALIGN;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessGlobalMemoryCacheType) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_device_mem_cache_type_t));

  ur_device_mem_cache_type_t property_value =
      UR_DEVICE_MEM_CACHE_TYPE_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_GE(property_value, UR_DEVICE_MEM_CACHE_TYPE_NONE);
  ASSERT_LT(property_value, UR_DEVICE_MEM_CACHE_TYPE_FORCE_UINT32);
}

TEST_P(urDeviceGetInfoTest, SuccessGlobalMemoryCacheLineSize) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessGlobalMemoryCacheSize) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint64_t));

  uint64_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessGlobalMemorySize) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_GLOBAL_MEM_SIZE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint64_t));

  uint64_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessGlobalMemoryFreeSize) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_GLOBAL_MEM_FREE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint64_t));

  // Free memory can be any value (even 0), but we assume that UR will not
  // run in environments with devices with 18EiB of memory
  uint64_t property_value = std::numeric_limits<uint64_t>::max();
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));
  ASSERT_NE(property_value, std::numeric_limits<uint64_t>::max());
}

TEST_P(urDeviceGetInfoTest, SuccessMaxConstantBufferSize) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint64_t));

  uint64_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMaxConstantArgs) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MAX_CONSTANT_ARGS;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessLocalMemoryType) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_LOCAL_MEM_TYPE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_device_local_mem_type_t));

  ur_device_local_mem_type_t property_value =
      UR_DEVICE_LOCAL_MEM_TYPE_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_GE(property_value, UR_DEVICE_LOCAL_MEM_TYPE_NONE);
  ASSERT_LT(property_value, UR_DEVICE_LOCAL_MEM_TYPE_FORCE_UINT32);
}

TEST_P(urDeviceGetInfoTest, SuccessLocalMemorySize) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_LOCAL_MEM_SIZE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint64_t));

  uint64_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessErrorCorrectionSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_ERROR_CORRECTION_SUPPORT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessHostUnifiedMemory) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_HOST_UNIFIED_MEMORY;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessProfilingTimerResolution) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_PROFILING_TIMER_RESOLUTION;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(size_t));

  size_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessEndianLittle) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_ENDIAN_LITTLE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessAvailable) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_AVAILABLE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessCompilerAvailable) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_COMPILER_AVAILABLE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessLinkerAvailable) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_LINKER_AVAILABLE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessExecutionCapabilities) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_EXECUTION_CAPABILITIES;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_device_exec_capability_flags_t));

  ur_device_exec_capability_flags_t property_value =
      UR_DEVICE_EXEC_CAPABILITY_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value & UR_DEVICE_EXEC_CAPABILITY_FLAGS_MASK, 0);
}

TEST_P(urDeviceGetInfoTest, SuccessQueueOnDeviceProperties) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_queue_flags_t));

  ur_queue_flags_t property_value = UR_QUEUE_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value & UR_QUEUE_FLAGS_MASK, 0);
}

TEST_P(urDeviceGetInfoTest, SuccessQueueOnHostProperties) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_queue_flags_t));

  ur_queue_flags_t property_value = UR_QUEUE_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value & UR_QUEUE_FLAGS_MASK, 0);
}

TEST_P(urDeviceGetInfoTest, SuccessBuiltInKernels) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_BUILT_IN_KERNELS;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_GT(property_size, 0);

  std::vector<char> property_value(property_size, '\0');
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 property_value.data(), nullptr));

  ASSERT_TRUE(uur::stringPropertyIsValid(property_value.data(), property_size));
}

TEST_P(urDeviceGetInfoTest, SuccessPlatform) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_PLATFORM;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_platform_handle_t));

  ur_platform_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value, platform);
}

TEST_P(urDeviceGetInfoTest, SuccessReferenceCount) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_REFERENCE_COUNT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessILVersion) {
  UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_IL_VERSION;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_GT(property_size, 0);

  std::vector<char> property_value(property_size, '\0');
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 property_value.data(), nullptr));
  ASSERT_TRUE(uur::stringPropertyIsValid(property_value.data(), property_size));
}

TEST_P(urDeviceGetInfoTest, SuccessName) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_NAME;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_GT(property_size, 0);

  std::vector<char> property_value(property_size, '\0');
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 property_value.data(), nullptr));
  ASSERT_TRUE(uur::stringPropertyIsValid(property_value.data(), property_size));
}

TEST_P(urDeviceGetInfoTest, SuccessVendor) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_VENDOR;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_GT(property_size, 0);

  std::vector<char> property_value(property_size, '\0');
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 property_value.data(), nullptr));
  ASSERT_TRUE(uur::stringPropertyIsValid(property_value.data(), property_size));
}

TEST_P(urDeviceGetInfoTest, SuccessDriverVersion) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_DRIVER_VERSION;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_GT(property_size, 0);

  std::vector<char> property_value(property_size, '\0');
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 property_value.data(), nullptr));
  ASSERT_TRUE(uur::stringPropertyIsValid(property_value.data(), property_size));
}

TEST_P(urDeviceGetInfoTest, SuccessProfile) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_PROFILE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_GT(property_size, 0);

  std::vector<char> property_value(property_size, '\0');
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 property_value.data(), nullptr));
  ASSERT_TRUE(uur::stringPropertyIsValid(property_value.data(), property_size));
}

TEST_P(urDeviceGetInfoTest, SuccessVersion) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_VERSION;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_GT(property_size, 0);

  std::vector<char> property_value(property_size, '\0');
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 property_value.data(), nullptr));
  ASSERT_TRUE(uur::stringPropertyIsValid(property_value.data(), property_size));
}

TEST_P(urDeviceGetInfoTest, SuccessBackendRuntimeVersion) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_BACKEND_RUNTIME_VERSION;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_GT(property_size, 0);

  std::vector<char> property_value(property_size, '\0');
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 property_value.data(), nullptr));
  ASSERT_TRUE(uur::stringPropertyIsValid(property_value.data(), property_size));
}

TEST_P(urDeviceGetInfoTest, SuccessExtensions) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_EXTENSIONS;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_GT(property_size, 0);

  std::vector<char> property_value(property_size, '\0');
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 property_value.data(), nullptr));
  ASSERT_TRUE(uur::stringPropertyIsValid(property_value.data(), property_size));
}

TEST_P(urDeviceGetInfoTest, SuccessPrintfBufferSize) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_PRINTF_BUFFER_SIZE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(size_t));

  size_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessPreferredInteropUserSync) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessParentDevice) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_PARENT_DEVICE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_device_handle_t));
}

TEST_P(urDeviceGetInfoTest, SuccessSupportedPartitions) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_SUPPORTED_PARTITIONS;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);

  if (property_size > 0) {
    size_t num_partitions = property_size / sizeof(ur_device_partition_t);
    std::vector<ur_device_partition_t> partitions(num_partitions);
    ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                   partitions.data(), nullptr));

    for (const auto &partition : partitions) {
      EXPECT_GE(partition, UR_DEVICE_PARTITION_EQUALLY);
      EXPECT_LT(partition, UR_DEVICE_PARTITION_FORCE_UINT32);
    }
  } else {
    ASSERT_EQ(property_size, 0);
  }
}

TEST_P(urDeviceGetInfoTest, SuccessPartitionMaxSubDevices) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessPartitionAffinityDomain) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_device_affinity_domain_flags_t));

  ur_device_affinity_domain_flags_t property_value =
      UR_DEVICE_AFFINITY_DOMAIN_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value & UR_DEVICE_AFFINITY_DOMAIN_FLAGS_MASK, 0);
}

TEST_P(urDeviceGetInfoTest, SuccessPartitionType) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_PARTITION_TYPE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);

  if (property_size > 0) {
    size_t num_properties =
        property_size / sizeof(ur_device_partition_property_t);
    std::vector<ur_device_partition_property_t> properties(num_properties);

    ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                   properties.data(), nullptr));

    for (const auto prop : properties) {
      ASSERT_GE(prop.type, UR_DEVICE_PARTITION_EQUALLY);
      ASSERT_LE(prop.type, UR_DEVICE_PARTITION_FORCE_UINT32);
    }
  } else {
    ASSERT_EQ(property_size, 0);
  }
}

TEST_P(urDeviceGetInfoTest, SuccessMaxSubGroups) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MAX_NUM_SUB_GROUPS;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessSubGroupIndependentForwardProgress) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessSubGroupSizesIntel) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL;

  ASSERT_SUCCESS(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));

  ASSERT_EQ(property_size % sizeof(uint32_t), 0);
}

TEST_P(urDeviceGetInfoTest, SuccessUSMHostSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_USM_HOST_SUPPORT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_device_usm_access_capability_flags_t));

  ur_device_usm_access_capability_flags_t property_value =
      UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAGS_MASK, 0);
}

TEST_P(urDeviceGetInfoTest, SuccessUSMDeviceSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_USM_DEVICE_SUPPORT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_device_usm_access_capability_flags_t));

  ur_device_usm_access_capability_flags_t property_value =
      UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAGS_MASK, 0);
}

TEST_P(urDeviceGetInfoTest, SuccessUSMSingleSharedSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_device_usm_access_capability_flags_t));

  ur_device_usm_access_capability_flags_t property_value =
      UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAGS_MASK, 0);
}

TEST_P(urDeviceGetInfoTest, SuccessUSMCrossSharedSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_device_usm_access_capability_flags_t));

  ur_device_usm_access_capability_flags_t property_value =
      UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAGS_MASK, 0);
}

TEST_P(urDeviceGetInfoTest, SuccessUSMSystemSharedSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_device_usm_access_capability_flags_t));

  ur_device_usm_access_capability_flags_t property_value =
      UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAGS_MASK, 0);
}

TEST_P(urDeviceGetInfoTest, SuccessUUID) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_UUID;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_GT(property_size, 0);

  std::vector<uint8_t> property_value(property_size, '\0');
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 property_value.data(), nullptr));
  EXPECT_NE(property_value[0], '\0');
}

TEST_P(urDeviceGetInfoTest, SuccessPCIAddress) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_PCI_ADDRESS;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_GT(property_size, 0);

  std::vector<char> property_value(property_size, '\0');
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 property_value.data(), nullptr));
  ASSERT_TRUE(uur::stringPropertyIsValid(property_value.data(), property_size));
}

TEST_P(urDeviceGetInfoTest, SuccessIntelGPUEUCount) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_GPU_EU_COUNT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessIntelGPUEUSIMDWidth) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_GPU_EU_SIMD_WIDTH;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessIntelGPUEUSlices) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_GPU_EU_SLICES;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessIntelGPUEUCountPerSlice) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessIntelGPUSubslicesPerSplice) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessIntelGPUHWThreadsPerEU) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_GPU_HW_THREADS_PER_EU;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessIntelMaxMemoryBandwidth) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MAX_MEMORY_BANDWIDTH;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);

  uint64_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessImageSRGB) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_IMAGE_SRGB;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessAtomic64) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_ATOMIC_64;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessAtomicMemoryOrderCapabilities) {
  size_t property_size = 0;
  ur_device_info_t property_name =
      UR_DEVICE_INFO_ATOMIC_MEMORY_ORDER_CAPABILITIES;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_memory_order_capability_flags_t));

  ur_memory_order_capability_flags_t property_value =
      UR_MEMORY_ORDER_CAPABILITY_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value & UR_MEMORY_ORDER_CAPABILITY_FLAGS_MASK, 0);
}

TEST_P(urDeviceGetInfoTest, SuccessAtomicMemoryScopeCapabilities) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});
  size_t property_size = 0;
  ur_device_info_t property_name =
      UR_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_memory_scope_capability_flags_t));

  ur_memory_scope_capability_flags_t property_value =
      UR_MEMORY_SCOPE_CAPABILITY_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value & UR_MEMORY_SCOPE_CAPABILITY_FLAGS_MASK, 0);
}

TEST_P(urDeviceGetInfoTest, SuccessAtomicFenceOrderCapabilities) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_ATOMIC_FENCE_ORDER_CAPABILITIES;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_memory_order_capability_flags_t));

  ur_memory_order_capability_flags_t property_value =
      UR_MEMORY_ORDER_CAPABILITY_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value & UR_MEMORY_ORDER_CAPABILITY_FLAGS_MASK, 0);
}

TEST_P(urDeviceGetInfoTest, SuccessAtomicFenceScopeCapabilities) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_ATOMIC_FENCE_SCOPE_CAPABILITIES;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_memory_scope_capability_flags_t));

  ur_memory_scope_capability_flags_t property_value =
      UR_MEMORY_SCOPE_CAPABILITY_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value & UR_MEMORY_SCOPE_CAPABILITY_FLAGS_MASK, 0);
}

TEST_P(urDeviceGetInfoTest, SuccessMaxComputeQueueIndices) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  size_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessKernelSetSpecializationConstants) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessMemoryBusWidth) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MEMORY_BUS_WIDTH;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMaxWorkGroups3D) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MAX_WORK_GROUPS_3D;

  ASSERT_SUCCESS(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(size_t) * 3);

  std::array<size_t, 3> max_work_group_sizes = {};
  ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_MAX_WORK_GROUPS_3D,
                                 sizeof(max_work_group_sizes),
                                 max_work_group_sizes.data(), nullptr));
  for (size_t i = 0; i < 3; i++) {
    ASSERT_NE(max_work_group_sizes[i], 0);
  }
}

TEST_P(urDeviceGetInfoTest, SuccessAsyncBarrier) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_ASYNC_BARRIER;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessMemoryChannelSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MEM_CHANNEL_SUPPORT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessHostPipeReadWriteSupport) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_HOST_PIPE_READ_WRITE_SUPPORT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessMaxRegistersPerWorkGroup) {
  UUR_KNOWN_FAILURE_ON(uur::NativeCPU{});

  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_MAX_REGISTERS_PER_WORK_GROUP;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessIPVersion) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_IP_VERSION;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessVirtualMemorySupported) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessESIMDSupported) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_ESIMD_SUPPORT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessCompositeDevice) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_COMPOSITE_DEVICE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      UR_DEVICE_INFO_COMPOSITE_DEVICE);

  ASSERT_EQ(property_size, sizeof(ur_device_handle_t));
}

TEST_P(urDeviceGetInfoTest, SuccessGlobalVariableSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_GLOBAL_VARIABLE_SUPPORT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessUSMPoolSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_USM_POOL_SUPPORT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessCommandBufferSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_COMMAND_BUFFER_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessCommandBufferUpdateCapabilities) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_COMMAND_BUFFER_UPDATE_CAPABILITIES_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size,
            sizeof(ur_device_command_buffer_update_capability_flags_t));

  ur_device_command_buffer_update_capability_flags_t property_value =
      UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value &
                UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAGS_MASK,
            0);
}

TEST_P(urDeviceGetInfoTest, SuccessCommandBufferEventSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_COMMAND_BUFFER_EVENT_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessBindlessImagesSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_BINDLESS_IMAGES_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessBindlessImagesSharedUSMSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_BINDLESS_IMAGES_SHARED_USM_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessBindlessImagesShared1DUSMSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_BINDLESS_IMAGES_1D_USM_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessBindlessImagesShared2DUSMSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_BINDLESS_IMAGES_2D_USM_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessImagePitchAlign) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_IMAGE_PITCH_ALIGN_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMaxImageLinearWidth) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_MAX_IMAGE_LINEAR_WIDTH_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(size_t));

  size_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMaxImageLinearHeight) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_MAX_IMAGE_LINEAR_HEIGHT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(size_t));

  size_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMaxImageLinearPitch) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_MAX_IMAGE_LINEAR_PITCH_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(size_t));

  size_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMipMapSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MIPMAP_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessMipMapAnisotropySupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_MIPMAP_ANISOTROPY_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessMipMapMaxAnisotropy) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_MIPMAP_MAX_ANISOTROPY_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(uint32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMipMapLevelReferenceSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_MIPMAP_LEVEL_REFERENCE_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessExternalMemoryImportSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_EXTERNAL_MEMORY_IMPORT_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessExternalSemaphoreImportSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_EXTERNAL_SEMAPHORE_IMPORT_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessCubemapSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_CUBEMAP_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessCubemapSeamlessFilteringSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_CUBEMAP_SEAMLESS_FILTERING_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessBindlessSampledImageFetch1DUSM) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_1D_USM_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessBindlessSampledImageFetch1D) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_1D_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessBindlessSampledImageFetch2DUSM) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_2D_USM_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessBindlessSampledImageFetch2D) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_2D_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessBindlessSampledImageFetch3D) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_BINDLESS_SAMPLED_IMAGE_FETCH_3D_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessTimestampRecordingSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_TIMESTAMP_RECORDING_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessImageArraySupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_IMAGE_ARRAY_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessBindlessUniqueAddressingPerDim) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_BINDLESS_UNIQUE_ADDRESSING_PER_DIM_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessBindlessSample1DUSM) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_BINDLESS_SAMPLE_1D_USM_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessBindlessSample2DUSM) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_BINDLESS_SAMPLE_2D_USM_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, SuccessEnqueueNativeCommandSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_ENQUEUE_NATIVE_COMMAND_SUPPORT_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  ur_bool_t property_value = false;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  bool casted_value = static_cast<bool>(property_value);
  ASSERT_TRUE(casted_value == false || casted_value == true);
}

TEST_P(urDeviceGetInfoTest, Success2DBlockArrayCapabilities) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_2D_BLOCK_ARRAY_CAPABILITIES_EXP;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size,
            sizeof(ur_exp_device_2d_block_array_capability_flags_t));

  ur_exp_device_2d_block_array_capability_flags_t property_value =
      UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value &
                UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAGS_MASK,
            0);
}

TEST_P(urDeviceGetInfoTest, SuccessUseNativeAssert) {
  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_USE_NATIVE_ASSERT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessBfloat16ConversionsNative) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_BFLOAT16_CONVERSIONS_NATIVE;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_bool_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessThrottleReasons) {
  // TODO: enable when driver/library version mismatch is fixed in CI.
  // See https://github.com/intel/llvm/issues/17614
  UUR_KNOWN_FAILURE_ON(uur::CUDA{});

  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_CURRENT_CLOCK_THROTTLE_REASONS;

  UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size));
  ASSERT_EQ(property_size, sizeof(ur_device_throttle_reasons_flag_t));

  ur_device_throttle_reasons_flag_t property_value =
      UR_DEVICE_THROTTLE_REASONS_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value & UR_DEVICE_THROTTLE_REASONS_FLAGS_MASK, 0);
}

TEST_P(urDeviceGetInfoTest, SuccessFanSpeed) {
  // TODO: enable when driver/library version mismatch is fixed in CI.
  // See https://github.com/intel/llvm/issues/17614
  UUR_KNOWN_FAILURE_ON(uur::CUDA{});

  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_FAN_SPEED;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);

  ASSERT_EQ(property_size, sizeof(int32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMaxPowerLimit) {
  // TODO: enable when driver/library version mismatch is fixed in CI.
  // See https://github.com/intel/llvm/issues/17614
  UUR_KNOWN_FAILURE_ON(uur::CUDA{});

  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MAX_POWER_LIMIT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);

  ASSERT_EQ(property_size, sizeof(int32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, SuccessMinPowerLimit) {
  // TODO: enable when driver/library version mismatch is fixed in CI.
  // See https://github.com/intel/llvm/issues/17614
  UUR_KNOWN_FAILURE_ON(uur::CUDA{});

  size_t property_size = 0;
  const ur_device_info_t property_name = UR_DEVICE_INFO_MIN_POWER_LIMIT;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);

  ASSERT_EQ(property_size, sizeof(int32_t));

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urDeviceGetInfo(device, property_name,
                                             property_size, &property_value,
                                             nullptr),
                             property_value);
}

TEST_P(urDeviceGetInfoTest, InvalidNullHandleDevice) {
  ur_device_type_t device_type;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urDeviceGetInfo(nullptr, UR_DEVICE_INFO_TYPE,
                                   sizeof(ur_device_type_t), &device_type,
                                   nullptr));
}

TEST_P(urDeviceGetInfoTest, InvalidEnumerationInfoType) {
  ur_device_type_t device_type;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urDeviceGetInfo(device, UR_DEVICE_INFO_FORCE_UINT32,
                                   sizeof(ur_device_type_t), &device_type,
                                   nullptr));
}

TEST_P(urDeviceGetInfoTest, InvalidSizePropSize) {
  ur_device_type_t device_type;
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_SIZE,
      urDeviceGetInfo(device, UR_DEVICE_INFO_TYPE, 0, &device_type, nullptr));
}

TEST_P(urDeviceGetInfoTest, InvalidSizePropSizeSmall) {
  ur_device_type_t device_type;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urDeviceGetInfo(device, UR_DEVICE_INFO_TYPE,
                                   sizeof(device_type) - 1, &device_type,
                                   nullptr));
}

TEST_P(urDeviceGetInfoTest, InvalidNullPointerPropValue) {
  ur_device_type_t device_type;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urDeviceGetInfo(device, UR_DEVICE_INFO_TYPE,
                                   sizeof(device_type), nullptr, nullptr));
}

TEST_P(urDeviceGetInfoTest, InvalidNullPointerPropSizeRet) {
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_POINTER,
      urDeviceGetInfo(device, UR_DEVICE_INFO_TYPE, 0, nullptr, nullptr));
}

using urDeviceGetInfoComponentDevicesTest = uur::urAllDevicesTest;
UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urDeviceGetInfoComponentDevicesTest);

TEST_P(urDeviceGetInfoComponentDevicesTest, SuccessComponentDevices) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  const ur_device_info_t property_name = UR_DEVICE_INFO_COMPONENT_DEVICES;

  for (const ur_device_handle_t device : devices) {
    ASSERT_NE(device, nullptr);

    ur_bool_t isComposite = false;
    ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
        urDeviceGetInfo(device, UR_DEVICE_INFO_COMPOSITE_DEVICE,
                        sizeof(ur_device_handle_t), &isComposite, nullptr),
        UR_DEVICE_INFO_COMPOSITE_DEVICE);

    if (isComposite) {
      size_t size = 0;
      ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
          urDeviceGetInfo(device, property_name, 0, nullptr, &size),
          property_name);

      size_t numComponents = size / sizeof(ur_device_handle_t);
      ASSERT_EQ(size % sizeof(ur_device_handle_t), 0);
      ASSERT_GT(numComponents, 0);

      std::vector<ur_device_handle_t> componentDevices(numComponents);
      ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, size,
                                     componentDevices.data(), nullptr));

      for (const ur_device_handle_t componentDevice : componentDevices) {
        ASSERT_NE(componentDevice, nullptr);
      }
    } else {
      size_t property_size = 999;

      ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
          urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
          property_name);
      ASSERT_EQ(property_size, 0);

      std::vector<ur_device_handle_t> componentDevices(property_size);
      ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                     componentDevices.data(), nullptr));

      ASSERT_TRUE(componentDevices.empty());
    }
  }
}

TEST_P(urDeviceGetInfoTest, SuccessKernelLaunchPropertiesSupport) {
  size_t property_size = 0;
  const ur_device_info_t property_name =
      UR_DEVICE_INFO_KERNEL_LAUNCH_CAPABILITIES;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urDeviceGetInfo(device, property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(property_size, sizeof(ur_kernel_launch_properties_flags_t));

  ur_kernel_launch_properties_flags_t property_value =
      UR_KERNEL_LAUNCH_PROPERTIES_FLAG_FORCE_UINT32;
  ASSERT_SUCCESS(urDeviceGetInfo(device, property_name, property_size,
                                 &property_value, nullptr));

  ASSERT_EQ(property_value & UR_KERNEL_LAUNCH_PROPERTIES_FLAGS_MASK, 0);
}
