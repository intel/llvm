// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>
#include <uur/known_failure.h>

using urPhysicalMemGetInfoTest = uur::urPhysicalMemTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urPhysicalMemGetInfoTest);

TEST_P(urPhysicalMemGetInfoTest, SuccessContext) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  ur_physical_mem_info_t property_name = UR_PHYSICAL_MEM_INFO_CONTEXT;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urPhysicalMemGetInfo(physical_mem, property_name, 0, nullptr,
                           &property_size),
      property_name);
  ASSERT_NE(property_size, 0);

  ur_context_handle_t returned_context = nullptr;
  ASSERT_SUCCESS(urPhysicalMemGetInfo(
      physical_mem, property_name, property_size, &returned_context, nullptr));

  ASSERT_EQ(context, returned_context);
}

TEST_P(urPhysicalMemGetInfoTest, SuccessDevice) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  ur_physical_mem_info_t property_name = UR_PHYSICAL_MEM_INFO_DEVICE;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urPhysicalMemGetInfo(physical_mem, property_name, 0, nullptr,
                           &property_size),
      property_name);
  ASSERT_NE(property_size, 0);

  ur_device_handle_t returned_device = nullptr;
  ASSERT_SUCCESS(urPhysicalMemGetInfo(
      physical_mem, property_name, property_size, &returned_device, nullptr));

  ASSERT_EQ(device, returned_device);
}

TEST_P(urPhysicalMemGetInfoTest, SuccessSize) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  ur_physical_mem_info_t property_name = UR_PHYSICAL_MEM_INFO_SIZE;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urPhysicalMemGetInfo(physical_mem, property_name, 0, nullptr,
                           &property_size),
      property_name);
  ASSERT_NE(property_size, 0);

  size_t returned_size = 0;
  ASSERT_SUCCESS(urPhysicalMemGetInfo(physical_mem, property_name,
                                      property_size, &returned_size, nullptr));

  ASSERT_EQ(size, returned_size);
}

TEST_P(urPhysicalMemGetInfoTest, SuccessProperties) {
  UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

  ur_physical_mem_info_t property_name = UR_PHYSICAL_MEM_INFO_PROPERTIES;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urPhysicalMemGetInfo(physical_mem, property_name, 0, nullptr,
                           &property_size),
      property_name);
  ASSERT_NE(property_size, 0);

  ur_physical_mem_properties_t returned_properties = {};
  ASSERT_SUCCESS(urPhysicalMemGetInfo(physical_mem, property_name,
                                      property_size, &returned_properties,
                                      nullptr));

  ASSERT_EQ(properties.stype, returned_properties.stype);
  ASSERT_EQ(properties.pNext, returned_properties.pNext);
  ASSERT_EQ(properties.flags, returned_properties.flags);
}

TEST_P(urPhysicalMemGetInfoTest, SuccessReferenceCount) {
  ur_physical_mem_info_t property_name = UR_PHYSICAL_MEM_INFO_REFERENCE_COUNT;
  size_t property_size = 0;

  ASSERT_SUCCESS_OR_OPTIONAL_QUERY(
      urPhysicalMemGetInfo(physical_mem, property_name, 0, nullptr,
                           &property_size),
      property_name);
  ASSERT_NE(property_size, 0);

  uint32_t returned_reference_count = 0;
  ASSERT_SUCCESS(urPhysicalMemGetInfo(physical_mem, property_name,
                                      property_size, &returned_reference_count,
                                      nullptr));

  ASSERT_EQ(returned_reference_count, 1);
}
