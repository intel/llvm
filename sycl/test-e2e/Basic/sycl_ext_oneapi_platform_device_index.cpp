// RUN: %{build} %level_zero_options %opencl_lib -o %t.out
// RUN: %{run} %t.out

#include "../helpers.hpp"

#include <level_zero/ze_api.h>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/platform.hpp>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>

int main() {
  sycl::device orig_dev;
  auto plt = orig_dev.get_platform();
  auto devices = plt.get_devices();
  auto it = std::find(devices.begin(), devices.end(), orig_dev);
  auto orig_dev_index_within_plt = std::distance(devices.begin(), it);

  // ext_oneapi_index_within_platform
  size_t ext_oneapi_index_within_platform =
      orig_dev.ext_oneapi_index_within_platform();

  // sycl_ext_oneapi_platform_device_index guarantees:
  // The device index returned from device::ext_oneapi_index_within_platform is
  // compatible with the index of the underlying backend device when the
  // ONEAPI_DEVICE_SELECTOR environment variable is not set.
  //
  // When the platform’s backend is backend::ext_oneapi_level_zero, the index
  // returned from device::ext_oneapi_index_within_platform matches the index of
  // the device’s underlying ze_device_handle_t within the list of handles
  // returned from zeDeviceGet.

  // When the platform’s backend is backend::opencl, the index returned from
  // device::ext_oneapi_index_within_platform matches the index of the device’s
  // underlying cl_device_id within the list of IDs returned from
  // clGetDeviceIDs.

  // Check if the index matches the index of the device’s underlying handle
  // within the list of handles returned from zeDeviceGet/clGetDeviceIDs.
  const char *selector = env::getVal("ONEAPI_DEVICE_SELECTOR");
  if (selector) {
    assert(orig_dev_index_within_plt == ext_oneapi_index_within_platform &&
           "The index returned from device::ext_oneapi_index_within_platform "
           "doesn't match the index that the device has in the std::vector "
           "that is returned when calling platform::get_devices() on the "
           "platform that contains this device");
  } else {
    if (orig_dev.get_backend() == sycl::backend::ext_oneapi_level_zero) {
      return 1;
      auto l0_plt = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(plt);
      auto l0_dev =
          sycl::get_native<sycl::backend::ext_oneapi_level_zero>(orig_dev);

      uint32_t num_devices = 0;
      zeDeviceGet(l0_plt, &num_devices, nullptr);

      std::vector<ze_device_handle_t> l0_devices(num_devices);
      zeDeviceGet(l0_plt, &num_devices, l0_devices.data());

      auto it = std::find(l0_devices.begin(), l0_devices.end(), l0_dev);
      assert(ext_oneapi_index_within_platform ==
                 (std::distance(l0_devices.begin(), it)) &&
             "The index returned from device::ext_oneapi_index_within_platform "
             "doesn't match the index of the device’s underlying cl_device_id "
             "within the list of IDs returned from clGetDeviceIDs");

    } else if (orig_dev.get_backend() == sycl::backend::opencl) {
      auto cl_plt = sycl::get_native<sycl::backend::opencl>(plt);
      auto cl_dev = sycl::get_native<sycl::backend::opencl>(orig_dev);

      cl_uint num_devices = 0;
      clGetDeviceIDs(cl_plt, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);

      std::vector<cl_device_id> cl_devices(num_devices);
      clGetDeviceIDs(cl_plt, CL_DEVICE_TYPE_ALL, num_devices, cl_devices.data(),
                     nullptr);

      auto it = std::find(cl_devices.begin(), cl_devices.end(), cl_dev);
      assert(orig_dev_index_within_plt ==
                 (std::distance(cl_devices.begin(), it)) &&
             "The index returned from device::ext_oneapi_index_within_platform "
             "doesn't match the index of the device’s underlying cl_device_id "
             "within the list of IDs returned from clGetDeviceIDs");
    }
  }
  // Test non-root device exception (if partition is supported)
  auto partition_properties =
      orig_dev.get_info<sycl::info::device::partition_properties>();
  if (std::find(partition_properties.begin(), partition_properties.end(),
                sycl::info::partition_property::partition_equally) !=
      partition_properties.end()) {
    std::vector<sycl::device> sub_devices = orig_dev.create_sub_devices<
        sycl::info::partition_property::partition_equally>(2);
    try {
      sub_devices[0].ext_oneapi_index_within_platform();
      assert(false && "Missing an exception");
    } catch (sycl::exception &e) {
      std::cout << e.what() << std::endl;
    }
  }

  // ext_oneapi_device_at_index
  auto ext_oneapi_device_at_index =
      plt.ext_oneapi_device_at_index(orig_dev_index_within_plt);
  assert(orig_dev == ext_oneapi_device_at_index &&
         "A copy of the device object which has that index doesn't match the "
         "original device");
  // Test out-of-range exception
  try {
    plt.ext_oneapi_device_at_index(devices.size());
    assert(false && "Missing an exception");
  } catch (sycl::exception &e) {
    std::cout << e.what() << std::endl;
  }

  return 0;
}
