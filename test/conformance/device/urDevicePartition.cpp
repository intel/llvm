// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urDevicePartitionTest = uur::urAllDevicesTest;

TEST_F(urDevicePartitionTest, PartitionEquallySuccess) {
  for (auto device : devices) {

    if (!uur::hasDevicePartitionSupport(
            device, UR_DEVICE_PARTITION_PROPERTY_FLAG_EQUALLY)) {
      GTEST_SKIP();
    }

    // get the number of compute units
    uint32_t n_compute_units = 0;
    ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_MAX_COMPUTE_UNITS,
                                   sizeof(n_compute_units), &n_compute_units,
                                   nullptr));
    ASSERT_NE(n_compute_units, 0);

    for (uint32_t i = 1; i < n_compute_units; ++i) {
      // TODO - I don't think the API is clear here as to how we should
      // terminate the array of property values. The spec says "null
      // terminated", which doesn't mean much for array of structs???
      // Additionally - if we want to use BY_COUNTS we need to supply an array
      // of counts of Compute Units to partition. This is not really possible
      // with the property_value struct here.
      ur_device_partition_property_value_t properties[] = {
          {UR_DEVICE_PARTITION_PROPERTY_FLAG_EQUALLY, i}, {}};

      // Get the number of devices that will be created
      uint32_t n_devices;
      ASSERT_SUCCESS(
          urDevicePartition(device, properties, 0, nullptr, &n_devices));
      ASSERT_NE(n_devices, 0);

      std::vector<ur_device_handle_t> sub_devices(n_devices);
      ASSERT_SUCCESS(urDevicePartition(
          device, properties, static_cast<uint32_t>(sub_devices.size()),
          sub_devices.data(), nullptr));
      for (auto sub_device : sub_devices) {
        ASSERT_NE(sub_device, nullptr);
        ASSERT_SUCCESS(urDeviceRelease(sub_device));
      }
    }
  }
}

// TODO - adds tests for BY_COUNTS & BY_AFFINITY_DOMAIN - #169
