// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %{build} -o %t.out %level_zero_options
// RUN: %{run} %t.out

// This test checks that an interop Level Zero device is properly handled during
// interop context construction.
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include <level_zero/ze_api.h>

#include <cassert>
#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
  int level0DriverIndex = 0;
  int level0DeviceIndex = 0;

  zeInit(0);
  uint32_t level0NumDrivers = 0;
  zeDriverGet(&level0NumDrivers, nullptr);

  assert(level0NumDrivers > 0);

  std::vector<ze_driver_handle_t> level0Drivers(level0NumDrivers);
  zeDriverGet(&level0NumDrivers, level0Drivers.data());

  ze_driver_handle_t level0Driver = level0Drivers[level0DriverIndex];
  uint32_t level0NumDevices = 0;
  zeDeviceGet(level0Driver, &level0NumDevices, nullptr);

  assert(level0NumDevices > 0);

  std::vector<ze_device_handle_t> level0Devices(level0NumDevices);
  zeDeviceGet(level0Driver, &level0NumDevices, level0Devices.data());

  ze_device_handle_t level0Device = level0Devices[level0DeviceIndex];
  ze_context_handle_t level0Context = nullptr;
  ze_context_desc_t level0ContextDesc = {};
  level0ContextDesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  zeContextCreateEx(level0Driver, &level0ContextDesc, 1, &level0Device,
                    &level0Context);

  sycl::device dev;
  sycl::device interopDev =
      sycl::make_device<sycl::backend::ext_oneapi_level_zero>(level0Device);
  sycl::context interopCtx =
      sycl::make_context<sycl::backend::ext_oneapi_level_zero>(
          {level0Context,
           {interopDev},
           sycl::ext::oneapi::level_zero::ownership::keep});

  assert(interopCtx.get_devices().size() == 1);
  assert(interopCtx.get_devices()[0] == interopDev);
  sycl::queue q{interopCtx, interopDev};

  zeContextDestroy(level0Context);
  return 0;
}
