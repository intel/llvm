// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out

#include <algorithm>
#include <iostream>
#include <level_zero/ze_api.h>
#include <stdio.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/detail/core.hpp>

using namespace sycl::ext::oneapi;

/*
  The purpose of this test is to verify the expected behvior for
  creating/caching platform and device objects. The available platforms and
  devices (returned from get_platforms() and get_devices()) should remain
  consistent throughout the lifetime of the application. Any new platform or
  device queried should be one that is a part of the original list of platforms
  and devices generated from get_platforms() and get_devices(). This test will
  verify that in the following ways:
    - Query for a list of all of the available platforms and devices
    - Create a Queue using a default device selector and verify that the device
      and platform associated with the queue can be found in the original
      devices and platforms lists.
    - Create both a platform and a device object using a native handle, and
      verify that they are in the original platform and device lists.
  This test will fail if it tries to create a platform or device that was not
  in the list of platforms and devices queried in the begininning.
*/

// Query the device and platform from a queue (constructed with a default device
// selector) and check if they can be found in the devices and platforms lists
// that were generated above.
int queryFromQueue(std::vector<sycl::platform> *platform_list,
                   std::vector<sycl::device> *device_list) {
  int failures = 0;
  sycl::queue Q{sycl::default_selector_v};
  sycl::device dev = Q.get_info<sycl::info::queue::device>();
  auto plt = dev.get_platform();

  std::cout << "Platform queried from Queue : "
            << plt.get_info<sycl::info::platform::name>() << std::endl;
  auto plt_result = std::find_if(platform_list->begin(), platform_list->end(),
                                 [&](sycl::platform &p) { return p == plt; });
  if (plt_result != platform_list->end()) {
    std::cout << "The platform list contains: "
              << plt.get_info<sycl::info::platform::name>() << std::endl;
  } else {
    std::cout << plt.get_info<sycl::info::platform::name>()
              << " was not in the platform list.\n";
    failures++;
  }

  std::cout << "Device queried from Queue : "
            << plt.get_info<sycl::info::platform::name>() << std::endl;
  auto dev_result = std::find_if(device_list->begin(), device_list->end(),
                                 [&](sycl::device &d) { return d == dev; });
  if (dev_result != device_list->end()) {
    std::cout << "The device list contains: "
              << dev.get_info<sycl::info::device::name>() << std::endl;
  } else {
    std::cout << dev.get_info<sycl::info::device::name>()
              << " was not in the device list.\n";
    failures++;
  }
  return failures;
}

// Create both a platform and a device object using a native handle, and check
// if those are in the platform and device lists.
int queryFromNativeHandle(std::vector<sycl::platform> *platform_list,
                          std::vector<sycl::device> *device_list) {
  int failures = 0;
  uint32_t l0_driver_count = 0;
  zeDriverGet(&l0_driver_count, nullptr);
  if (l0_driver_count == 0) {
    std::cout << "There is no Level Zero Driver available\n";
    return failures;
  }
  std::vector<ze_driver_handle_t> l0_drivers(l0_driver_count);
  zeDriverGet(&l0_driver_count, l0_drivers.data());

  uint32_t l0_device_count = 0;
  zeDeviceGet(l0_drivers[0], &l0_device_count, nullptr);
  if (l0_device_count == 0) {
    std::cout << "There is no Level Zero Device available\n";
    return failures;
  }
  std::vector<ze_device_handle_t> l0_devices(l0_device_count);
  zeDeviceGet(l0_drivers[0], &l0_device_count, l0_devices.data());

  // Create the platform and device objects using the native handle.
  auto plt =
      sycl::make_platform<sycl::backend::ext_oneapi_level_zero>(l0_drivers[0]);
  auto dev =
      sycl::make_device<sycl::backend::ext_oneapi_level_zero>(l0_devices[0]);

  // Check to see if this platform is in the platform list.
  std::cout << "Platform created with native handle: "
            << plt.get_info<sycl::info::platform::name>() << std::endl;
  auto plt_result = std::find_if(platform_list->begin(), platform_list->end(),
                                 [&](sycl::platform &p) { return p == plt; });
  if (plt_result != platform_list->end()) {
    std::cout << "The platform list contains: "
              << plt.get_info<sycl::info::platform::name>() << std::endl;
  } else {
    std::cout << plt.get_info<sycl::info::platform::name>()
              << " was not in the platform list.\n";
    failures++;
  }

  // Check to see if this device is in the device list.
  std::cout << "Device created with native handle: "
            << dev.get_info<sycl::info::device::name>() << std::endl;
  auto dev_result = std::find_if(device_list->begin(), device_list->end(),
                                 [&](sycl::device &d) { return d == dev; });
  if (dev_result != device_list->end()) {
    std::cout << "The device list contains: "
              << dev.get_info<sycl::info::device::name>() << std::endl;
    auto dev_result = std::find_if(device_list->begin(), device_list->end(),
                                   [&](sycl::device &d) { return d == dev; });
    if (dev_result != device_list->end()) {
      // Level-Zero backend specification for sycl::make_device:
      //
      //   > Constructs a SYCL device instance from a Level-Zero
      //   > ze_device_handle_t. The SYCL execution environment for the Level
      //   > Zero backend contains a fixed number of devices that are enumerated
      //   > via sycl::device::get_devices() and a fixed number of sub-devices
      //   > that are enumerated via sycl::device::create_sub_devices(...).
      //   > Calling this function does not create a new device. Rather it
      //   > merely creates a sycl::device object that is a copy of one of the
      //   > devices from those enumerations.
      //
      // SYCL 2020's common reference semantics says that such a copy must
      // result in the same hash value.
      auto hash = std::hash<sycl::device>{};
      assert(hash(*dev_result) == hash(dev));

      std::cout << "The device list contains: "
                << dev.get_info<sycl::info::device::name>() << std::endl;
    } else {
      std::cout << dev.get_info<sycl::info::device::name>()
                << " was not in the device list.\n";
      failures++;
    }
  }

  return failures;
}

int main() {
  int failures = 0;

  // Query for a list of all of the available platforms and devices.
  int pindex = 1;
  std::vector<sycl::platform> platform_list;
  std::vector<sycl::device> device_list;
  for (const auto &plt : sycl::platform::get_platforms()) {
    std::cout << "Platform " << pindex++ << " "
              << " (" << plt.get_info<sycl::info::platform::name>() << ")"
              << std::endl;
    platform_list.push_back(plt);

    int dindex = 1;
    for (const auto &dev : plt.get_devices()) {
      std::cout << "  "
                << "Device " << dindex++ << " ("
                << dev.get_info<sycl::info::device::name>() << ")" << std::endl;
      device_list.push_back(dev);
    }
  }

  failures = queryFromQueue(&platform_list, &device_list);
  failures += queryFromNativeHandle(&platform_list, &device_list);

  return failures;
}
