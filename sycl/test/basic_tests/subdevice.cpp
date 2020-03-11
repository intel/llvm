// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==------------ subdevice.cpp - SYCL subdevice basic test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../helpers.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
#include <utility>

using namespace cl::sycl;

int main() {
  // Not catching exceptions to make test fail instead.

  auto devices = device::get_devices();
  for (const auto &dev : devices) {
    // TODO: implement subdevices creation for host device
    if (dev.is_host())
      continue;

    CHECK(dev.get_info<info::device::partition_type_property>() ==
          info::partition_property::no_partition);

    size_t MaxSubDevices =
        dev.get_info<info::device::partition_max_sub_devices>();
    if (MaxSubDevices == 0)
      continue;

    try {
      auto SubDevicesEq =
          dev.create_sub_devices<info::partition_property::partition_equally>(
              1);
      CHECK(SubDevicesEq.size() == MaxSubDevices &&
            "Requested 1 compute unit in each subdevice, expected maximum "
            "number of subdevices in output");
      std::cout << "Created " << SubDevicesEq.size()
                << " subdevices using equal partition scheme" << std::endl;

      CHECK(SubDevicesEq[0].get_info<info::device::partition_type_property>() ==
            info::partition_property::partition_equally);

      CHECK(SubDevicesEq[0].get_info<info::device::parent_device>().get() ==
            dev.get());
    } catch (feature_not_supported) {
      // okay skip it
    }

    try {
      vector_class<size_t> Counts(MaxSubDevices, 1);
      auto SubDevicesByCount = dev.create_sub_devices<
          info::partition_property::partition_by_counts>(Counts);
      CHECK(SubDevicesByCount.size() == MaxSubDevices &&
            "Maximum number of subdevices was requested with 1 compute unit "
            "on each");
      std::cout << "Created " << SubDevicesByCount.size()
                << " subdevices using partition by counts scheme."
                << std::endl;
      CHECK(SubDevicesByCount[0]
                .get_info<info::device::partition_type_property>() ==
            info::partition_property::partition_by_counts);
    } catch (feature_not_supported) {
      // okay skip it
    }

    try {
      auto SubDevicesDomainNuma = dev.create_sub_devices<
          info::partition_property::partition_by_affinity_domain>(
          info::partition_affinity_domain::numa);
      std::cout
          << "Created " << SubDevicesDomainNuma.size()
          << " subdevices using partition by numa affinity domain scheme."
          << std::endl;
    } catch (feature_not_supported) {
      // okay skip it
    }

    try {
      auto SubDevicesDomainL4 = dev.create_sub_devices<
          info::partition_property::partition_by_affinity_domain>(
          info::partition_affinity_domain::L4_cache);
      std::cout << "Created " << SubDevicesDomainL4.size()
                << " subdevices using partition by L4 cache domain scheme."
                << std::endl;
    } catch (feature_not_supported) {
      // okay skip it
    }

    try {
      auto SubDevicesDomainL3 = dev.create_sub_devices<
          info::partition_property::partition_by_affinity_domain>(
          info::partition_affinity_domain::L3_cache);
      std::cout << "Created " << SubDevicesDomainL3.size()
                << " subdevices using partition by L3 cache domain scheme."
                << std::endl;
    } catch (feature_not_supported) {
      // okay skip it
    }

    try {
      auto SubDevicesDomainL2 = dev.create_sub_devices<
          info::partition_property::partition_by_affinity_domain>(
          info::partition_affinity_domain::L2_cache);
      std::cout << "Created " << SubDevicesDomainL2.size()
                << " subdevices using partition by L2 cache domain scheme."
                << std::endl;
    } catch (feature_not_supported) {
      // okay skip it
    }

    try {
      auto SubDevicesDomainL1 = dev.create_sub_devices<
          info::partition_property::partition_by_affinity_domain>(
          info::partition_affinity_domain::L1_cache);
      std::cout << "Created " << SubDevicesDomainL1.size()
                << " subdevices using partition by L1 cache domain scheme."
                << std::endl;
    } catch (feature_not_supported) {
      // okay skip it
    }

    try {
      auto SubDevicesDomainNextPart = dev.create_sub_devices<
          info::partition_property::partition_by_affinity_domain>(
          info::partition_affinity_domain::next_partitionable);
      std::cout << "Created " << SubDevicesDomainNextPart.size()
                << " subdevices using partition by next partitionable "
                   "domain scheme."
                << std::endl;
    } catch (feature_not_supported) {
      // okay skip it
    }
  }

  return 0;
}
