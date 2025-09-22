// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/fixtures.h"
#include "uur/raii.h"

using urContextCreateTest = uur::urDeviceTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urContextCreateTest);

TEST_P(urContextCreateTest, Success) {
  uur::raii::Context context = nullptr;
  ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, context.ptr()));
  ASSERT_NE(nullptr, context);
}

TEST_P(urContextCreateTest, SuccessWithProperties) {
  ur_context_properties_t properties{UR_STRUCTURE_TYPE_CONTEXT_PROPERTIES,
                                     nullptr, 0};
  uur::raii::Context context = nullptr;
  ASSERT_SUCCESS(urContextCreate(1, &device, &properties, context.ptr()));
  ASSERT_NE(nullptr, context);
}

TEST_P(urContextCreateTest, InvalidNullPointerDevices) {
  uur::raii::Context context = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urContextCreate(1, nullptr, nullptr, context.ptr()));
}

TEST_P(urContextCreateTest, InvalidNullPointerContext) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urContextCreate(1, &device, nullptr, nullptr));
}

TEST_P(urContextCreateTest, InvalidEnumeration) {
  ur_context_properties_t properties{UR_STRUCTURE_TYPE_CONTEXT_PROPERTIES,
                                     nullptr, UR_CONTEXT_FLAGS_MASK};
  uur::raii::Context context = nullptr;

  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urContextCreate(1, &device, &properties, context.ptr()));
}

TEST_P(urContextCreateTest, SuccessParentAndSubDevices) {
  if (!uur::hasDevicePartitionSupport(device,
                                      UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN)) {
    GTEST_SKIP() << "Device \'" << device
                 << "\' does not support partitioning by affinity domain.\n";
  }

  ur_device_affinity_domain_flags_t flag = UR_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA;
  ur_device_affinity_domain_flags_t supported_flags{0};
  ASSERT_SUCCESS(
      uur::GetDevicePartitionAffinityDomainFlags(device, supported_flags));
  if (!(flag & supported_flags)) {
    GTEST_SKIP() << static_cast<ur_device_affinity_domain_flag_t>(flag)
                 << " is not supported by the device: \'" << device << "\'.\n";
  }

  ur_device_partition_property_t prop =
      uur::makePartitionByAffinityDomain(flag);

  ur_device_partition_properties_t properties{
      UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES,
      nullptr,
      &prop,
      1,
  };

  // Get the number of devices that will be created
  uint32_t n_devices = 0;
  ASSERT_SUCCESS(
      urDevicePartition(device, &properties, 0, nullptr, &n_devices));
  ASSERT_NE(n_devices, 0);

  std::vector<ur_device_handle_t> sub_devices(n_devices);
  ASSERT_SUCCESS(urDevicePartition(device, &properties,
                                   static_cast<uint32_t>(sub_devices.size()),
                                   sub_devices.data(), nullptr));

  std::vector<ur_device_handle_t> all_devices;
  all_devices.push_back(device);
  all_devices.insert(all_devices.end(), sub_devices.begin(), sub_devices.end());
  uur::raii::Context context = nullptr;
  ASSERT_SUCCESS(urContextCreate(static_cast<uint32_t>(all_devices.size()),
                                 all_devices.data(), nullptr, context.ptr()));
  ASSERT_NE(nullptr, context);

  for (auto sub_device : sub_devices) {
    ASSERT_NE(sub_device, nullptr);
    ASSERT_SUCCESS(urDeviceRelease(sub_device));
  }
}

using urContextCreateMultiDeviceTest = uur::urAllDevicesTest;
UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urContextCreateMultiDeviceTest);

TEST_P(urContextCreateMultiDeviceTest, Success) {
  if (devices.size() < 2) {
    GTEST_SKIP();
  }
  uur::raii::Context context = nullptr;
  ASSERT_SUCCESS(urContextCreate(static_cast<uint32_t>(devices.size()),
                                 devices.data(), nullptr, context.ptr()));
  ASSERT_NE(nullptr, context);
}
