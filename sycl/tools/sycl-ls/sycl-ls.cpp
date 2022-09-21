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
#include <sycl/sycl.hpp>

#include <cstdlib>
#include <iostream>
#include <map>
#include <stdlib.h>

using namespace sycl;

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

std::string getDeviceTypeName(const device &Device) {
  auto DeviceType = Device.get_info<info::device::device_type>();
  switch (DeviceType) {
  case info::device_type::cpu:
    return "cpu";
  case info::device_type::gpu:
    return "gpu";
  case info::device_type::host:
    return "host";
  case info::device_type::accelerator:
    return "acc";
  default:
    return "unknown";
  }
}

static void printDeviceInfo(const device &Device, bool Verbose,
                            const std::string &Prepend) {
  auto DeviceVersion = Device.get_info<info::device::version>();
  auto DeviceName = Device.get_info<info::device::name>();
  auto DeviceVendor = Device.get_info<info::device::vendor>();
  auto DeviceDriverVersion = Device.get_info<info::device::driver_version>();

  if (Verbose) {
    std::cout << Prepend << "Type       : " << getDeviceTypeName(Device)
              << std::endl;
    std::cout << Prepend << "Version    : " << DeviceVersion << std::endl;
    std::cout << Prepend << "Name       : " << DeviceName << std::endl;
    std::cout << Prepend << "Vendor     : " << DeviceVendor << std::endl;
    std::cout << Prepend << "Driver     : " << DeviceDriverVersion << std::endl;
  } else {
    std::cout << Prepend << ", " << DeviceName << " " << DeviceVersion << " ["
              << DeviceDriverVersion << "]" << std::endl;
  }
}

static void printSelectorChoice(const device_selector &Selector,
                                const std::string &Prepend) {
  try {
    const auto &Device = device(Selector);
    std::string DeviceTypeName = getDeviceTypeName(Device);
    auto Platform = Device.get_info<info::device::platform>();
    auto PlatformName = Platform.get_info<info::platform::name>();
    printDeviceInfo(Device, verbose,
                    Prepend + DeviceTypeName + ", " + PlatformName);
  } catch (const sycl::runtime_error &Exception) {
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

  const char *filter = std::getenv("SYCL_DEVICE_FILTER");
  if (filter) {
    std::cerr << "Warning: SYCL_DEVICE_FILTER environment variable is set to "
              << filter << "." << std::endl;
    std::cerr
        << "To see the correct device id, please unset SYCL_DEVICE_FILTER."
        << std::endl
        << std::endl;
  }

  const auto &Platforms = platform::get_platforms();

  // Keep track of the number of devices per backend
  std::map<backend, size_t> DeviceNums;

  for (const auto &Platform : Platforms) {
    backend Backend = Platform.get_backend();
    auto PlatformName = Platform.get_info<info::platform::name>();
    const auto &Devices = Platform.get_devices();
    for (const auto &Device : Devices) {
      std::cout << "[" << Backend << ":" << getDeviceTypeName(Device) << ":"
                << DeviceNums[Backend] << "] ";
      ++DeviceNums[Backend];
      // Verbose parameter is set to false to print regular devices output first
      printDeviceInfo(Device, false, PlatformName);
    }
  }

  if (verbose) {
    std::cout << "\nPlatforms: " << Platforms.size() << std::endl;
    uint32_t PlatformNum = 0;
    DeviceNums.clear();
    for (const auto &Platform : Platforms) {
      backend Backend = Platform.get_backend();
      ++PlatformNum;
      auto PlatformVersion = Platform.get_info<info::platform::version>();
      auto PlatformName = Platform.get_info<info::platform::name>();
      auto PlatformVendor = Platform.get_info<info::platform::vendor>();
      std::cout << "Platform [#" << PlatformNum << "]:" << std::endl;
      std::cout << "    Version  : " << PlatformVersion << std::endl;
      std::cout << "    Name     : " << PlatformName << std::endl;
      std::cout << "    Vendor   : " << PlatformVendor << std::endl;

      const auto &Devices = Platform.get_devices();
      std::cout << "    Devices  : " << Devices.size() << std::endl;
      for (const auto &Device : Devices) {
        std::cout << "        Device [#" << DeviceNums[Backend]
                  << "]:" << std::endl;
        ++DeviceNums[Backend];
        printDeviceInfo(Device, true, "        ");
      }
    }
  } else {
    return EXIT_SUCCESS;
  }

  // Print the selectors choice in one-line always
  verbose = false;

  // Print built-in device selectors choice
  printSelectorChoice(default_selector(), "default_selector()      : ");
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
