// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: MIT
#include <uur/fixtures.h>

using urDevicePartitionTest = uur::urAllDevicesTest;

TEST_F(urDevicePartitionTest, PartitionEquallySuccess) {
  for (auto device : devices) {

    if (!uur::hasDevicePartitionSupport(device, UR_DEVICE_PARTITION_EQUALLY)) {
      GTEST_SKIP();
    }

    // get the number of compute units
    uint32_t n_compute_units = 0;
    ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_MAX_COMPUTE_UNITS,
                                   sizeof(n_compute_units), &n_compute_units,
                                   nullptr));
    ASSERT_NE(n_compute_units, 0);

    for (uint32_t i = 1; i < n_compute_units; ++i) {
      ur_device_partition_property_t properties[] = {
          UR_DEVICE_PARTITION_EQUALLY, i, 0};

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
