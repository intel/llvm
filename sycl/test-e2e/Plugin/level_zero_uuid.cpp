// REQUIRES: gpu, level_zero, level_zero_dev_kit

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER

// Test that the UUID is read correctly from Level Zero.

// CHECK: PASSED
#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <level_zero/ze_api.h>
#include <sstream>

int main() {
  sycl::device dev;
  if (dev.has(sycl::aspect::ext_intel_device_info_uuid)) {
    auto uuid = dev.get_info<sycl::ext::intel::info::device::uuid>();
    std::stringstream uuid_sycl;
    for (int i = 0; i < uuid.size(); ++i)
      uuid_sycl << std::hex << std::setw(2) << std::setfill('0')
                << int(uuid[i]);
    std::cout << "SYCL: " << uuid_sycl.str() << std::endl;

    auto zedev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
    ze_device_properties_t device_properties{};
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
  }
  return 0;
}
