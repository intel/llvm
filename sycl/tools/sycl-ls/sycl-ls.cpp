//==----------- sycl-ls.cpp ------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The "sycl-ls" utility lists all platforms/devices discovered by SYCL similar
// to how clinfo prints this for OpenCL devices.
//
// There are two types of output:
//   concise (default) and
//   verbose (enabled with --verbose).
//
// In verbose mode it also prints, which devices would be chosen by various SYCL
// device selectors.
//
#include <CL/sycl.hpp>

#include <cstdlib>
#include <iostream>
#include <stdlib.h>

using namespace cl::sycl;

// Controls verbose output vs. concise.
bool verbose;

// Trivial custom selector that selects a device of the given type.
class custom_selector : public device_selector {
  info::device_type MType;

public:
  custom_selector(info::device_type Type) : MType(Type) {}
  int operator()(const device &Dev) const override {
    return Dev.get_info<info::device::device_type>() == MType ? 1 : -1;
  }
};

static void printDeviceInfo(const device &Device, const std::string &Prepend) {
  auto DeviceType = Device.get_info<info::device::device_type>();
  std::string DeviceTypeName;
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
    DeviceTypeName = "UNKNOWN";
    break;
  }

  auto DeviceVersion = Device.get_info<info::device::version>();
  auto DeviceName = Device.get_info<info::device::name>();
  auto DeviceVendor = Device.get_info<info::device::vendor>();
  auto DeviceDriverVersion = Device.get_info<info::device::driver_version>();

  if (verbose) {
    std::cout << Prepend << "Type       : " << DeviceTypeName << std::endl;
    std::cout << Prepend << "Version    : " << DeviceVersion << std::endl;
    std::cout << Prepend << "Name       : " << DeviceName << std::endl;
    std::cout << Prepend << "Vendor     : " << DeviceVendor << std::endl;
    std::cout << Prepend << "Driver     : " << DeviceDriverVersion << std::endl;
  } else {
    auto DevicePlatform = Device.get_info<info::device::platform>();
    auto DevicePlatformName = DevicePlatform.get_info<info::platform::name>();
    std::cout << Prepend << DeviceTypeName << ": " << DevicePlatformName << " "
              << DeviceVersion << " [" << DeviceDriverVersion << "]"
              << std::endl;
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
    std::cout << Prepend << What << std::endl;
  }
}

int main(int argc, char **argv) {

  // See if verbose output is requested
  if (argc == 1)
    verbose = false;
  else if (argc == 2 && std::string(argv[1]) == "--verbose")
    verbose = true;
  else {
    std::cout << "Usage: sycl-ls [--verbose]" << std::endl;
    return EXIT_FAILURE;
  }

  const auto &Platforms = platform::get_platforms();
  if (verbose)
    std::cout << "Platforms: " << Platforms.size() << std::endl;

  uint32_t PlatformNum = 0;

  for (const auto &Platform : Platforms) {
    uint32_t DeviceNum = 0;
    ++PlatformNum;
    if (verbose) {
      auto PlatformVersion = Platform.get_info<info::platform::version>();
      auto PlatformName = Platform.get_info<info::platform::name>();
      auto PlatformVendor = Platform.get_info<info::platform::vendor>();
      std::cout << "Platform [#" << PlatformNum << "]:" << std::endl;
      std::cout << "    Version  : " << PlatformVersion << std::endl;
      std::cout << "    Name     : " << PlatformName << std::endl;
      std::cout << "    Vendor   : " << PlatformVendor << std::endl;
    }
    const auto &Devices = Platform.get_devices();
    if (verbose)
      std::cout << "    Devices  : " << Devices.size() << std::endl;
    for (const auto &Device : Devices) {
      if (verbose)
        std::cout << "        Device [#" << DeviceNum << "]:" << std::endl;
      else {
        backend Backend = Platform.get_backend();
        std::cout << "[" << Backend << ":" << DeviceNum << "] ";
      }
      ++DeviceNum;
      printDeviceInfo(Device, verbose ? "        " : "");
    }
  }

  if (!verbose) {
    return EXIT_SUCCESS;
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

  return EXIT_SUCCESS;
}
