// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception
#include <uur/fixtures.h>

struct urVirtualMemGranularityGetInfoTest : uur::urContextTest {
  void SetUp() override {
    UUR_KNOWN_FAILURE_ON(uur::LevelZeroV2{});

    UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
    ur_bool_t virtual_memory_support = false;
    ASSERT_SUCCESS(
        urDeviceGetInfo(this->device, UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT,
                        sizeof(ur_bool_t), &virtual_memory_support, nullptr));
    if (!virtual_memory_support) {
      GTEST_SKIP() << "Virtual memory is not supported.";
    }
  }
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urVirtualMemGranularityGetInfoTest);

TEST_P(urVirtualMemGranularityGetInfoTest, SuccessMinimum) {
  size_t property_size = 0;
  ur_virtual_mem_granularity_info_t property_name =
      UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urVirtualMemGranularityGetInfo(context, device, property_name, 0, nullptr,
                                     &property_size),
      property_name);
  ASSERT_EQ(sizeof(size_t), property_size);

  uint32_t returned_minimum = 0;
  ASSERT_SUCCESS(urVirtualMemGranularityGetInfo(context, device, property_name,
                                                property_size,
                                                &returned_minimum, nullptr));

  ASSERT_GT(returned_minimum, 0);
}

TEST_P(urVirtualMemGranularityGetInfoTest, SuccessRecommended) {
  size_t property_size = 0;
  ur_virtual_mem_granularity_info_t property_name =
      UR_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urVirtualMemGranularityGetInfo(context, device, property_name, 0, nullptr,
                                     &property_size),
      property_name);
  ASSERT_EQ(sizeof(size_t), property_size);

  uint32_t returned_recommended = 0;
  ASSERT_SUCCESS(urVirtualMemGranularityGetInfo(
      context, device, property_name, property_size, &returned_recommended,
      nullptr));

  ASSERT_GT(returned_recommended, 0);
}

TEST_P(urVirtualMemGranularityGetInfoTest, InvalidNullHandleContext) {
  size_t property_size = 0;
  ASSERT_EQ_RESULT(urVirtualMemGranularityGetInfo(
                       nullptr, device, UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM,
                       0, nullptr, &property_size),
                   UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urVirtualMemGranularityGetInfoTest, InvalidEnumeration) {
  size_t property_size = 0;
  ASSERT_EQ_RESULT(urVirtualMemGranularityGetInfo(
                       context, device,
                       UR_VIRTUAL_MEM_GRANULARITY_INFO_FORCE_UINT32, 0, nullptr,
                       &property_size),
                   UR_RESULT_ERROR_INVALID_ENUMERATION);
}

TEST_P(urVirtualMemGranularityGetInfoTest, InvalidNullPointerPropSizeRet) {
  ASSERT_EQ_RESULT(urVirtualMemGranularityGetInfo(
                       context, device, UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM,
                       0, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urVirtualMemGranularityGetInfoTest, InvalidNullPointerPropValue) {
  ASSERT_EQ_RESULT(urVirtualMemGranularityGetInfo(
                       context, device, UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM,
                       sizeof(size_t), nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urVirtualMemGranularityGetInfoTest, InvalidPropSizeZero) {
  size_t minimum = 0;
  ASSERT_EQ_RESULT(urVirtualMemGranularityGetInfo(
                       context, device, UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM,
                       0, &minimum, nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urVirtualMemGranularityGetInfoTest, InvalidSizePropSizeSmall) {
  size_t minimum = 0;
  ASSERT_EQ_RESULT(urVirtualMemGranularityGetInfo(
                       context, device, UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM,
                       sizeof(size_t) - 1, &minimum, nullptr),
                   UR_RESULT_ERROR_INVALID_SIZE);
}
