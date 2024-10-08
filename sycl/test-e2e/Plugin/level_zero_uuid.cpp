// REQUIRES: aspect-ext_intel_device_info_uuid
// REQUIRES: gpu, level_zero, level_zero_dev_kit

// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out 2>&1 | FileCheck %s

// Test that the UUID is read correctly from Level Zero.

// CHECK: PASSED
#include <iomanip>
#include <iostream>
#include <level_zero/ze_api.h>
#include <sstream>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>

int main() {
  sycl::device dev;
  auto uuid = dev.get_info<sycl::ext::intel::info::device::uuid>();
  std::stringstream uuid_sycl;
  for (int i = 0; i < uuid.size(); ++i)
    uuid_sycl << std::hex << std::setw(2) << std::setfill('0') << int(uuid[i]);
  std::cout << "SYCL: " << uuid_sycl.str() << std::endl;

  auto zedev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
  ze_device_properties_t device_properties{};
  device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  zeDeviceGetProperties(zedev, &device_properties);
  std::stringstream uuid_l0;
  for (int i = 0; i < ZE_MAX_DEVICE_UUID_SIZE; ++i)
    uuid_l0 << std::hex << std::setw(2) << std::setfill('0')
            << int(device_properties.uuid.id[i]);
  std::cout << "L0  : " << uuid_l0.str() << std::endl;

  if (uuid_sycl.str() != uuid_l0.str()) {
    std::cout << "FAILED" << std::endl;
    return -1;
  }
  std::cout << "PASSED" << std::endl;
  return 0;
}
