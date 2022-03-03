// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// Nvidia should not allow sub_devices but does not throw corresponding error.
// XFAIL: hip_nvidia
/* Check that:
1) [info::device::partition_properties]: returns the partition properties
supported by this SYCL device; a vector of info::partition_property. If this
SYCL device cannot be partitioned into at least two sub devices then the
returned vector **must be empty**.

2) [create_sub_devices()]: If the SYCL device
does not support info::partition_property::partition_by_affinity_domain or the
SYCL device does not support the info::partition_affinity_domain provided, an
exception with the **feature_not_supported error code must be thrown**.
*/

#include <CL/sycl.hpp>

/** returns true if the device supports a particular affinity domain
 */
static bool
supports_affinity_domain(const cl::sycl::device &dev,
                         cl::sycl::info::partition_property partitionProp,
                         cl::sycl::info::partition_affinity_domain domain) {
  if (partitionProp !=
      cl::sycl::info::partition_property::partition_by_affinity_domain) {
    return true;
  }
  auto supported =
      dev.get_info<cl::sycl::info::device::partition_affinity_domains>();
  for (cl::sycl::info::partition_affinity_domain dom : supported) {
    if (dom == domain) {
      return true;
    }
  }
  return false;
}

/** returns true if the device supports a particular partition property
 */
static bool
supports_partition_property(const cl::sycl::device &dev,
                            cl::sycl::info::partition_property partitionProp) {
  auto supported = dev.get_info<cl::sycl::info::device::partition_properties>();
  for (cl::sycl::info::partition_property prop : supported) {
    if (prop == partitionProp) {
      return true;
    }
  }
  return false;
}

int main() {

  auto dev = cl::sycl::device(cl::sycl::default_selector());

  cl::sycl::info::partition_property partitionProperty =
      cl::sycl::info::partition_property::partition_by_affinity_domain;
  cl::sycl::info::partition_affinity_domain affinityDomain =
      cl::sycl::info::partition_affinity_domain::next_partitionable;

  if (supports_partition_property(dev, partitionProperty)) {
    if (supports_affinity_domain(dev, partitionProperty, affinityDomain)) {
      auto subDevices = dev.create_sub_devices<
          cl::sycl::info::partition_property::partition_by_affinity_domain>(
          affinityDomain);

      if (subDevices.size() < 2) {
        std::cerr << "device::create_sub_device(info::partition_affinity_"
                     "domain) should have returned at least 2 devices"
                  << std::endl;
        return -1;
      }
    }
    auto maxUnits = dev.get_info<sycl::info::device::max_compute_units>();
    try {
      std::vector<sycl::device> v = dev.create_sub_devices<
          sycl::info::partition_property::partition_equally>(maxUnits + 1);
      std::cerr << "create_sub_devices with more than max_compute_units should "
                   "have thrown an error"
                << std::endl;
      return -1;
    } catch (sycl::exception &ex) {
      if (ex.code() != sycl::errc::invalid) {
        std::cerr << "while an exception was correctly thrown, it has the "
                     "wrong error code"
                     "we should have received"
                  << sycl::errc::invalid << "but instead got"
                  << ex.code().value() << std::endl;
        return -1;
      }
    }
  } else {
    try {
      auto subDevices = dev.create_sub_devices<
          cl::sycl::info::partition_property::partition_by_affinity_domain>(
          affinityDomain);
      std::cerr << "device::create_sub_device(info::partition_affinity_domain) "
                   "should have thrown an exception"
                << std::endl;
      return -1;
    } catch (const cl::sycl::feature_not_supported &e) {
      if (e.code() != sycl::errc::feature_not_supported) {
        std::cerr
            << "error code should be errc::feature_not_supported instead of "
            << e.code().value() << std::endl;
        return -1;
      }
    } catch (...) {
      std::cerr << "device::create_sub_device(info::partition_affinity_domain) "
                   "should have thrown cl::sycl::feature_not_supported"
                << std::endl;
      return -1;
    }
  }
  return 0;
}
