// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==------------ subdevice.cpp - SYCL subdevice basic test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cassert>
#include <iostream>
#include <sycl/backend.hpp>
#include <sycl/backend/opencl.hpp>
#include <sycl/detail/core.hpp>
#include <utility>

using namespace sycl;

int main() {
  try {
    auto devices = device::get_devices();
    for (const auto &dev : devices) {
      assert(dev.get_info<info::device::partition_type_property>() ==
             info::partition_property::no_partition);

      size_t MaxSubDevices =
          dev.get_info<info::device::partition_max_sub_devices>();
      if (MaxSubDevices == 0)
        continue;

      try {
        auto SubDevicesEq =
            dev.create_sub_devices<info::partition_property::partition_equally>(
                1);
        assert(SubDevicesEq.size() == MaxSubDevices &&
               "Requested 1 compute unit in each subdevice, expected maximum "
               "number of subdevices in output");
        std::cout << "Created " << SubDevicesEq.size()
                  << " subdevices using equal partition scheme" << std::endl;

        assert(
            SubDevicesEq[0].get_info<info::device::partition_type_property>() ==
            info::partition_property::partition_equally);

        assert(sycl::get_native<sycl::backend::opencl>(
                   SubDevicesEq[0].get_info<info::device::parent_device>()) ==
               sycl::get_native<sycl::backend::opencl>(dev));
      } catch (const sycl::exception &e) {
        if (e.code() != sycl::errc::feature_not_supported)
          std::rethrow_exception(std::current_exception());
        // otherwise okay skip it
      }

      try {
        std::vector<size_t> Counts(MaxSubDevices, 1);
        auto SubDevicesByCount = dev.create_sub_devices<
            info::partition_property::partition_by_counts>(Counts);
        assert(SubDevicesByCount.size() == MaxSubDevices &&
               "Maximum number of subdevices was requested with 1 compute unit "
               "on each");
        std::cout << "Created " << SubDevicesByCount.size()
                  << " subdevices using partition by counts scheme."
                  << std::endl;
        assert(SubDevicesByCount[0]
                   .get_info<info::device::partition_type_property>() ==
               info::partition_property::partition_by_counts);
      } catch (const sycl::exception &e) {
        if (e.code() != sycl::errc::feature_not_supported)
          std::rethrow_exception(std::current_exception());
        // otherwise okay skip it
      }

      try {
        auto SubDevicesDomainNuma = dev.create_sub_devices<
            info::partition_property::partition_by_affinity_domain>(
            info::partition_affinity_domain::numa);
        std::cout
            << "Created " << SubDevicesDomainNuma.size()
            << " subdevices using partition by numa affinity domain scheme."
            << std::endl;
      } catch (const sycl::exception &e) {
        if (e.code() != sycl::errc::feature_not_supported)
          std::rethrow_exception(std::current_exception());
        // otherwise okay skip it
      }

      try {
        auto SubDevicesDomainL4 = dev.create_sub_devices<
            info::partition_property::partition_by_affinity_domain>(
            info::partition_affinity_domain::L4_cache);
        std::cout << "Created " << SubDevicesDomainL4.size()
                  << " subdevices using partition by L4 cache domain scheme."
                  << std::endl;
      } catch (const sycl::exception &e) {
        if (e.code() != sycl::errc::feature_not_supported)
          std::rethrow_exception(std::current_exception());
        // otherwise okay skip it
      }

      try {
        auto SubDevicesDomainL3 = dev.create_sub_devices<
            info::partition_property::partition_by_affinity_domain>(
            info::partition_affinity_domain::L3_cache);
        std::cout << "Created " << SubDevicesDomainL3.size()
                  << " subdevices using partition by L3 cache domain scheme."
                  << std::endl;
      } catch (const sycl::exception &e) {
        if (e.code() != sycl::errc::feature_not_supported)
          std::rethrow_exception(std::current_exception());
        // otherwise okay skip it
      }

      try {
        auto SubDevicesDomainL2 = dev.create_sub_devices<
            info::partition_property::partition_by_affinity_domain>(
            info::partition_affinity_domain::L2_cache);
        std::cout << "Created " << SubDevicesDomainL2.size()
                  << " subdevices using partition by L2 cache domain scheme."
                  << std::endl;
      } catch (const sycl::exception &e) {
        if (e.code() != sycl::errc::feature_not_supported)
          std::rethrow_exception(std::current_exception());
        // otherwise okay skip it
      }

      try {
        auto SubDevicesDomainL1 = dev.create_sub_devices<
            info::partition_property::partition_by_affinity_domain>(
            info::partition_affinity_domain::L1_cache);
        std::cout << "Created " << SubDevicesDomainL1.size()
                  << " subdevices using partition by L1 cache domain scheme."
                  << std::endl;
      } catch (const sycl::exception &e) {
        if (e.code() != sycl::errc::feature_not_supported)
          std::rethrow_exception(std::current_exception());
        // otherwise okay skip it
      }

      try {
        auto SubDevicesDomainNextPart = dev.create_sub_devices<
            info::partition_property::partition_by_affinity_domain>(
            info::partition_affinity_domain::next_partitionable);
        std::cout << "Created " << SubDevicesDomainNextPart.size()
                  << " subdevices using partition by next partitionable "
                     "domain scheme."
                  << std::endl;
      } catch (const sycl::exception &e) {
        if (e.code() != sycl::errc::feature_not_supported)
          std::rethrow_exception(std::current_exception());
        // otherwise okay skip it
      }

      // test exception
      try {
        const size_t out_of_bounds = std::numeric_limits<size_t>::max();
        const auto partition =
            sycl::info::partition_property::partition_equally;
        dev.create_sub_devices<partition>(out_of_bounds);
        std::cout << "we should not be here. Exception not thrown."
                  << std::endl;
        return 1;
      } catch (sycl::exception &e) {
        const auto code = e.code();

        if (!(code == sycl::errc::feature_not_supported ||
              code == sycl::errc::invalid)) {
          std::cout << "SYCL exception has wrong error code: " << code
                    << std::endl;
          return 1;
        }
      } catch (...) {
        std::cout << "Something besides a sycl::exception was thrown."
                  << std::endl;
        return 1;
      }
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
