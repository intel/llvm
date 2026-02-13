// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>
#include <uur/known_failure.h>

using urPhysicalMemGetInfoTest = uur::urPhysicalMemTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urPhysicalMemGetInfoTest);

bool operator==(ur_physical_mem_properties_t lhs,
                ur_physical_mem_properties_t rhs) {
  return lhs.flags == rhs.flags && lhs.pNext == rhs.pNext &&
         lhs.stype == rhs.stype;
}

TEST_P(urPhysicalMemGetInfoTest, SuccessContext) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  const ur_physical_mem_info_t property_name = UR_PHYSICAL_MEM_INFO_CONTEXT;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urPhysicalMemGetInfo(physical_mem, property_name, 0, nullptr,
                           &property_size),
      property_name);
  ASSERT_NE(property_size, 0);

  ur_context_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urPhysicalMemGetInfo(physical_mem, property_name,
                                      property_size, &property_value, nullptr));

  ASSERT_EQ(context, property_value);
}

TEST_P(urPhysicalMemGetInfoTest, SuccessDevice) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  const ur_physical_mem_info_t property_name = UR_PHYSICAL_MEM_INFO_DEVICE;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urPhysicalMemGetInfo(physical_mem, property_name, 0, nullptr,
                           &property_size),
      property_name);
  ASSERT_NE(property_size, 0);

  ur_device_handle_t property_value = nullptr;
  ASSERT_SUCCESS(urPhysicalMemGetInfo(physical_mem, property_name,
                                      property_size, &property_value, nullptr));

  ASSERT_EQ(device, property_value);
}

TEST_P(urPhysicalMemGetInfoTest, SuccessSize) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  const ur_physical_mem_info_t property_name = UR_PHYSICAL_MEM_INFO_SIZE;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urPhysicalMemGetInfo(physical_mem, property_name, 0, nullptr,
                           &property_size),
      property_name);
  ASSERT_NE(property_size, 0);

  size_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urPhysicalMemGetInfo(physical_mem, property_name,
                                                  property_size,
                                                  &property_value, nullptr),
                             property_value);

  ASSERT_EQ(size, property_value);
}

TEST_P(urPhysicalMemGetInfoTest, SuccessProperties) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{}, uur::LevelZeroV2{});

  const ur_physical_mem_info_t property_name = UR_PHYSICAL_MEM_INFO_PROPERTIES;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urPhysicalMemGetInfo(physical_mem, property_name, 0, nullptr,
                           &property_size),
      property_name);
  ASSERT_NE(property_size, 0);

  ur_physical_mem_properties_t property_value = {};
  ASSERT_SUCCESS(urPhysicalMemGetInfo(physical_mem, property_name,
                                      property_size, &property_value, nullptr));

  ASSERT_EQ(properties, property_value);
}

TEST_P(urPhysicalMemGetInfoTest, SuccessReferenceCount) {
  const ur_physical_mem_info_t property_name =
      UR_PHYSICAL_MEM_INFO_REFERENCE_COUNT;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urPhysicalMemGetInfo(physical_mem, property_name, 0, nullptr,
                           &property_size),
      property_name);
  ASSERT_NE(property_size, 0);

  uint32_t property_value = 0;
  ASSERT_QUERY_RETURNS_VALUE(urPhysicalMemGetInfo(physical_mem, property_name,
                                                  property_size,
                                                  &property_value, nullptr),
                             property_value);

  ASSERT_EQ(property_value, 1);
}
