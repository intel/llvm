//==- fpga_device_selector.hpp --- SYCL FPGA device selector shortcut  -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/device.hpp>
#include <sycl/device_selector.hpp>

#include <string>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

// Forward declaration
class platform;

namespace ext {
namespace intel {

class platform_selector : public device_selector {
private:
  std::string device_platform_name;

public:
  platform_selector(const std::string &platform_name)
      : device_platform_name(platform_name) {}

  int operator()(const device &device) const override {
    const platform &pf = device.get_platform();
    const std::string &platform_name = pf.get_info<info::platform::name>();
    if (platform_name == device_platform_name) {
      const std::string name = device.get_info<sycl::info::device::name>();
      // Prefer hardware devices over simulator ones.
      if (name.find("SimulatorDevice") != std::string::npos)
        return 9000;
      return 10000;
    }
    return -1;
  }
};

static constexpr auto EMULATION_PLATFORM_NAME =
    "Intel(R) FPGA Emulation Platform for OpenCL(TM)";
static constexpr auto HARDWARE_PLATFORM_NAME =
    "Intel(R) FPGA SDK for OpenCL(TM)";

class fpga_selector : public platform_selector {
public:
  fpga_selector() : platform_selector(HARDWARE_PLATFORM_NAME) {}
};

class fpga_emulator_selector : public platform_selector {
public:
  fpga_emulator_selector() : platform_selector(EMULATION_PLATFORM_NAME) {}
};

class fpga_simulator_selector : public fpga_selector {
public:
  fpga_simulator_selector(unsigned int numDevices = 1) {
    // Tell the runtime to start a simulator device
    // Notes:
    //   This replaces any hardware devices present
    //   Currently only 1 simulator device is supported
    auto devices = std::to_string(numDevices);
#ifdef _WIN32
    _putenv_s("CL_CONTEXT_MPSIM_DEVICE_INTELFPGA", devices.c_str());
#else
    setenv("CL_CONTEXT_MPSIM_DEVICE_INTELFPGA", devices.c_str(), 0);
#endif
  }
};

} // namespace intel
} // namespace ext

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
