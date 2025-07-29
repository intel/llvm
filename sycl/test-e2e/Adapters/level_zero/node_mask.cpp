// REQUIRES: aspect-ext_intel_device_info_node_mask
// REQUIRES: gpu, level_zero, level_zero_dev_kit, windows

// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out

// Test that the node mask is read correctly from Level Zero.

#include <iomanip>
#include <iostream>
#include <level_zero/ze_api.h>
#include <sstream>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>

int main() {
  sycl::device dev;
  auto nodeMaskSYCL = dev.get_info<sycl::ext::intel::info::device::node_mask>();

  std::cout << "SYCL: " << nodeMaskSYCL << std::endl;

  auto zedev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
  ze_device_properties_t device_properties{};
  device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

  ze_device_luid_ext_properties_t luid_device_properties{};
  luid_device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES;

  device_properties.pNext = &luid_device_properties;

  zeDeviceGetProperties(zedev, &device_properties);

  ze_device_luid_ext_properties_t *luid_dev_prop =
      static_cast<ze_device_luid_ext_properties_t *>(device_properties.pNext);

  uint32_t nodeMaskL0 = luid_dev_prop->nodeMask;

  std::cout << "L0  : " << nodeMaskL0 << std::endl;

  if (nodeMaskSYCL != nodeMaskL0) {
    std::cout << "FAILED" << std::endl;
    return -1;
  }

  std::cout << "PASSED" << std::endl;
  return 0;
}
