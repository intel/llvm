// REQUIRES: aspect-ext_intel_device_info_luid
// REQUIRES: gpu, level_zero, level_zero_dev_kit, windows

// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out

// Test that the LUID is read correctly from Level Zero.

#include <iomanip>
#include <iostream>
#include <level_zero/ze_api.h>
#include <sstream>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>

int main() {
  sycl::device dev;
  auto luid = dev.get_info<sycl::ext::intel::info::device::luid>();

  std::stringstream luid_sycl;
  for (int i = 0; i < luid.size(); ++i) {
    luid_sycl << std::hex << std::setw(2) << std::setfill('0') << int(luid[i]);
  }
  std::cout << "SYCL: " << luid_sycl.str() << std::endl;

  auto zedev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
  ze_device_properties_t device_properties{};
  device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

  ze_device_luid_ext_properties_t luid_device_properties{};
  luid_device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES;

  device_properties.pNext = &luid_device_properties;

  zeDeviceGetProperties(zedev, &device_properties);

  ze_device_luid_ext_properties_t *luid_dev_prop =
      static_cast<ze_device_luid_ext_properties_t *>(device_properties.pNext);

  std::stringstream luid_l0;
  for (int i = 0; i < ZE_MAX_DEVICE_LUID_SIZE_EXT; ++i)
    luid_l0 << std::hex << std::setw(2) << std::setfill('0')
            << int(luid_dev_prop->luid.id[i]);
  std::cout << "L0  : " << luid_l0.str() << std::endl;

  if (luid_sycl.str() != luid_l0.str()) {
    std::cout << "FAILED" << std::endl;
    return -1;
  }

  std::cout << "PASSED" << std::endl;
  return 0;
}
