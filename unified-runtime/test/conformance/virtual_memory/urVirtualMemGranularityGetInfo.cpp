// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

struct urVirtualMemGranularityGetInfoTest : uur::urContextTest {
  void SetUp() override {
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

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urVirtualMemGranularityGetInfoTest);

void urVirtualMemGranularityGetInfoTest_successCase(
    ur_context_handle_t context, ur_device_handle_t device,
    const ur_virtual_mem_granularity_info_t property_name,
    const size_t allocation_size) {
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urVirtualMemGranularityGetInfo(context, device, allocation_size,
                                     property_name, 0, nullptr, &property_size),
      property_name);
  ASSERT_EQ(sizeof(size_t), property_size);

  size_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(
      urVirtualMemGranularityGetInfo(context, device, allocation_size,
                                     property_name, property_size,
                                     &property_value, nullptr),
      property_value);

  ASSERT_GT(property_value, 0);
}

TEST_P(urVirtualMemGranularityGetInfoTest, SuccessMinimum_smallAllocation) {
  urVirtualMemGranularityGetInfoTest_successCase(
      context, device, UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM, 1);
}

TEST_P(urVirtualMemGranularityGetInfoTest, SuccessMinimum_largeAllocation) {
  urVirtualMemGranularityGetInfoTest_successCase(
      context, device, UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM, 191439360);
}

TEST_P(urVirtualMemGranularityGetInfoTest, SuccessRecommended_smallAllocation) {
  urVirtualMemGranularityGetInfoTest_successCase(
      context, device, UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM, 19);
}

TEST_P(urVirtualMemGranularityGetInfoTest, SuccessRecommended_largeAllocation) {
  urVirtualMemGranularityGetInfoTest_successCase(
      context, device, UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM, 211739367);
}

TEST_P(urVirtualMemGranularityGetInfoTest, InvalidNullHandleContext) {
  size_t property_size = 0;
  ASSERT_EQ_RESULT(
      urVirtualMemGranularityGetInfo(nullptr, device, 1,
                                     UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM, 0,
                                     nullptr, &property_size),
      UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urVirtualMemGranularityGetInfoTest, InvalidEnumeration) {
  size_t property_size = 0;
  ASSERT_EQ_RESULT(urVirtualMemGranularityGetInfo(
                       context, device, 1,
                       UR_VIRTUAL_MEM_GRANULARITY_INFO_FORCE_UINT32, 0, nullptr,
                       &property_size),
                   UR_RESULT_ERROR_INVALID_ENUMERATION);
}

TEST_P(urVirtualMemGranularityGetInfoTest, InvalidNullPointerPropSizeRet) {
  ASSERT_EQ_RESULT(
      urVirtualMemGranularityGetInfo(context, device, 1,
                                     UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM, 0,
                                     nullptr, nullptr),
      UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urVirtualMemGranularityGetInfoTest, InvalidNullPointerPropValue) {
  ASSERT_EQ_RESULT(
      urVirtualMemGranularityGetInfo(context, device, 1,
                                     UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM,
                                     sizeof(size_t), nullptr, nullptr),
      UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urVirtualMemGranularityGetInfoTest, InvalidPropSizeZero) {
  size_t minimum = 0;
  ASSERT_EQ_RESULT(
      urVirtualMemGranularityGetInfo(context, device, 1,
                                     UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM, 0,
                                     &minimum, nullptr),
      UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urVirtualMemGranularityGetInfoTest, InvalidSizePropSizeSmall) {
  size_t minimum = 0;
  ASSERT_EQ_RESULT(
      urVirtualMemGranularityGetInfo(context, device, 1,
                                     UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM,
                                     sizeof(size_t) - 1, &minimum, nullptr),
      UR_RESULT_ERROR_INVALID_SIZE);
}
