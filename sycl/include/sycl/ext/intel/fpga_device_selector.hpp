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
    const std::string &platform_name =
        pf.get_info<sycl::info::platform::name>();
    if (platform_name == device_platform_name) {
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
  fpga_simulator_selector() {
    // Tell the runtime to use a simulator device rather than hardware
#ifdef _WIN32
    _putenv_s("CL_CONTEXT_MPSIM_DEVICE_INTELFPGA", "1");
#else
    setenv("CL_CONTEXT_MPSIM_DEVICE_INTELFPGA", "1", 0);
#endif
  }
};

} // namespace intel
} // namespace ext

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
