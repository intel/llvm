// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/known_failure.h>

struct urPhysicalMemCreateTest
    : uur::urVirtualMemGranularityTestWithParam<size_t> {

  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(
        uur::urVirtualMemGranularityTestWithParam<size_t>::SetUp());
    size = getParam() * granularity;
  }

  void TearDown() override {
    if (physical_mem) {
      ASSERT_SUCCESS(urPhysicalMemRelease(physical_mem));
    }
    uur::urVirtualMemGranularityTestWithParam<size_t>::TearDown();
  }

  size_t size;
  ur_physical_mem_handle_t physical_mem = nullptr;
};

using urPhysicalMemCreateWithSizeParamTest = urPhysicalMemCreateTest;
UUR_DEVICE_TEST_SUITE_WITH_PARAM(urPhysicalMemCreateWithSizeParamTest,
                                 ::testing::Values(1, 2, 3, 7, 12, 44),
                                 uur::deviceTestWithParamPrinter<size_t>);

TEST_P(urPhysicalMemCreateWithSizeParamTest, Success) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  ASSERT_SUCCESS(
      urPhysicalMemCreate(context, device, size, nullptr, &physical_mem));
  ASSERT_NE(physical_mem, nullptr);
}

TEST_P(urPhysicalMemCreateWithSizeParamTest, InvalidSize) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  if (granularity == 1) {
    GTEST_SKIP() << "A granularity of 1 means that any size will be accepted.";
  }
  size_t invalid_size = size - 1;
  ASSERT_EQ_RESULT(urPhysicalMemCreate(context, device, invalid_size, nullptr,
                                       &physical_mem),
                   UR_RESULT_ERROR_INVALID_SIZE);
}

using urPhysicalMemCreateWithFlagsParamTest =
    uur::urPhysicalMemTestWithParam<ur_physical_mem_flags_t>;
UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urPhysicalMemCreateWithFlagsParamTest,
    ::testing::Values(UR_PHYSICAL_MEM_FLAG_TBD),
    uur::deviceTestWithParamPrinter<ur_physical_mem_flags_t>);

TEST_P(urPhysicalMemCreateWithFlagsParamTest, Success) {
  ur_physical_mem_properties_t properties;
  properties.stype = UR_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES;
  properties.pNext = nullptr;
  properties.flags = getParam();

  ASSERT_SUCCESS(
      urPhysicalMemCreate(context, device, size, &properties, &physical_mem));
  ASSERT_NE(physical_mem, nullptr);
}

using urPhysicalMemCreateTest = urPhysicalMemCreateTest;
UUR_DEVICE_TEST_SUITE_WITH_PARAM(urPhysicalMemCreateTest, ::testing::Values(1),
                                 uur::deviceTestWithParamPrinter<size_t>);

TEST_P(urPhysicalMemCreateTest, InvalidNullHandleContext) {
  ASSERT_EQ_RESULT(
      urPhysicalMemCreate(nullptr, device, size, nullptr, &physical_mem),
      UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urPhysicalMemCreateTest, InvalidNullHandleDevice) {
  ASSERT_EQ_RESULT(
      urPhysicalMemCreate(context, nullptr, size, nullptr, &physical_mem),
      UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urPhysicalMemCreateTest, InvalidNullPointerPhysicalMem) {
  ASSERT_EQ_RESULT(urPhysicalMemCreate(context, device, size, nullptr, nullptr),
                   UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urPhysicalMemCreateTest, InvalidEnumeration) {
  ur_physical_mem_properties_t properties;
  properties.stype = UR_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES;
  properties.pNext = nullptr;
  properties.flags = UR_PHYSICAL_MEM_FLAG_FORCE_UINT32;

  ASSERT_EQ_RESULT(
      urPhysicalMemCreate(context, device, size, &properties, &physical_mem),
      UR_RESULT_ERROR_INVALID_ENUMERATION);
}
