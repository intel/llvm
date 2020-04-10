//==----------- lspi.cpp ---------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The "lspi" utility lists all platforms/devices discovered by PI similar to
// how lscl prints this for OpenCL devices. It can probably be eventually merged
// with the more complex "check-sycl" utility.
//
// There are two types of output:
//   concise (default) and
//   verbose (enabled with any argument).
//
// In verbose mode it also prints, which devices would be chose  by various SYCL
// device selectors.
//
#include <CL/sycl.hpp>

#include <cstdlib>
#include <iostream>
#include <stdlib.h>

using namespace cl::sycl;
using namespace std;

// Controls verbose output vs. concise.
bool verbose = false;

// Trivial custom selector that selects a device of the given type.
class custom_selector : public device_selector {
  info::device_type MType;

public:
  custom_selector(info::device_type Type) : MType(Type) {}
  int operator()(const device &Dev) const override {
    return Dev.get_info<info::device::device_type>() == MType ? 1 : -1;
  }
};

static void printDeviceInfo(const device &Device, const string &Prepend) {
  auto DeviceType = Device.get_info<info::device::device_type>();
  string DeviceTypeName;
  switch (DeviceType) {
  case info::device_type::cpu:
    DeviceTypeName = "CPU ";
    break;
  case info::device_type::gpu:
    DeviceTypeName = "GPU ";
    break;
  case info::device_type::host:
    DeviceTypeName = "HOST";
    break;
  case info::device_type::accelerator:
    DeviceTypeName = "ACC ";
    break;
  default:
    DeviceTypeName = "??? ";
    break;
  }

  auto DeviceVersion = Device.get_info<info::device::version>();
  auto DeviceName = Device.get_info<info::device::name>();
  auto DeviceVendor = Device.get_info<info::device::vendor>();
  auto DeviceDriverVersion = Device.get_info<info::device::driver_version>();

  if (verbose) {
    cout << Prepend << "Type       : " << DeviceTypeName << std::endl;
    cout << Prepend << "Version    : " << DeviceVersion << std::endl;
    cout << Prepend << "Name       : " << DeviceName << std::endl;
    cout << Prepend << "Vendor     : " << DeviceVendor << std::endl;
    cout << Prepend << "Driver     : " << DeviceDriverVersion << std::endl;
  } else {
    cout << Prepend << DeviceTypeName << ": " << DeviceVersion << "[ "
         << DeviceDriverVersion << " ]" << std::endl;
  }
}

static void printSelectorChoice(const device_selector &Selector,
                                const std::string &Prepend) {
  try {
    const auto &Dev = device(Selector);
    printDeviceInfo(Dev, Prepend);

  } catch (const cl::sycl::runtime_error &Exception) {
    // Truncate long string so it can fit in one-line
    std::string What = Exception.what();
    if (What.length() > 50)
      What = What.substr(0, 50) + "...";

    if (verbose) {
      cout << Prepend << "Type       : " << What << std::endl;
      cout << Prepend << "Version    : " << What << std::endl;
      cout << Prepend << "Name       : " << What << std::endl;
      cout << Prepend << "Vendor     : " << What << std::endl;
      cout << Prepend << "Driver     : " << What << std::endl;
    } else {
      cout << Prepend << What << std::endl;
    }
  }
}

int main(int argc, char **argv) {

  // Any options trigger verbose output.
  if (argc > 1) {
    verbose = true;
    if (!std::getenv("SYCL_PI_TRACE")) {
      // Enable trace of PI discovery.
      // setenv("SYCL_PI_TRACE", "1", true);
    }
  }

  auto Platforms = platform::get_platforms();
  if (verbose)
    cout << "Platforms: " << Platforms.size() << std::endl;

  uint32_t PlatformNum = 0;
  for (const auto &Platform : Platforms) {
    ++PlatformNum;
    if (verbose) {
      auto PlatformVersion = Platform.get_info<info::platform::version>();
      auto PlatformName = Platform.get_info<info::platform::name>();
      auto PlatformVendor = Platform.get_info<info::platform::vendor>();
      cout << "Platform [#" << PlatformNum << "]:" << std::endl;
      cout << "    Version  : " << PlatformVersion << std::endl;
      cout << "    Name     : " << PlatformName << std::endl;
      cout << "    Vendor   : " << PlatformVendor << std::endl;
    }
    auto Devices = Platform.get_devices();
    if (verbose)
      cout << "    Devices  : " << Devices.size() << std::endl;
    uint32_t DeviceNum = 0;
    for (const auto &Device : Devices) {
      ++DeviceNum;
      if (verbose)
        cout << "        Device [#" << DeviceNum << "]:" << std::endl;
      printDeviceInfo(Device, verbose ? "        " : "");
    }
  }

  if (!verbose) {
    return 0;
  }

  // Print the selectors choice in one-line always
  verbose = false;

  // Print built-in device selectors choice
  printSelectorChoice(default_selector(), "default_selector()      : ");
  printSelectorChoice(host_selector(), "host_selector()         : ");
  printSelectorChoice(accelerator_selector(), "accelerator_selector()  : ");
  printSelectorChoice(cpu_selector(), "cpu_selector()          : ");
  printSelectorChoice(gpu_selector(), "gpu_selector()          : ");

  // Print trivial custom selectors choice
  printSelectorChoice(custom_selector(info::device_type::gpu),
                      "custom_selector(gpu)    : ");
  printSelectorChoice(custom_selector(info::device_type::cpu),
                      "custom_selector(cpu)    : ");
  printSelectorChoice(custom_selector(info::device_type::accelerator),
                      "custom_selector(acc)    : ");

  return 0;
}
