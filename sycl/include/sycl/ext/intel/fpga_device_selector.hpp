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
#include <string_view>

namespace sycl {
inline namespace _V1 {

// Forward declaration
class platform;

namespace ext::intel {

namespace detail {
// Scores a device by platform name.
inline int selectDeviceByPlatform(std::string_view required_platform_name,
                                  const device &device) {
  if (device.get_platform().get_info<sycl::info::platform::name>() ==
      required_platform_name)
    return 10000;
  return -1;
}

// Enables an environment variable required by the FPGA simulator.
inline void enableFPGASimulator() {
#ifdef _WIN32
  _putenv_s("CL_CONTEXT_MPSIM_DEVICE_INTELFPGA", "1");
#else
  setenv("CL_CONTEXT_MPSIM_DEVICE_INTELFPGA", "1", 0);
#endif
}
} // namespace detail

class platform_selector : public device_selector {
private:
  std::string device_platform_name;

public:
  platform_selector(const std::string &platform_name)
      : device_platform_name(platform_name) {}

  int operator()(const device &device) const override {
    return detail::selectDeviceByPlatform(device_platform_name, device);
  }
};

static constexpr auto EMULATION_PLATFORM_NAME =
    "Intel(R) FPGA Emulation Platform for OpenCL(TM)";
static constexpr auto HARDWARE_PLATFORM_NAME =
    "Intel(R) FPGA SDK for OpenCL(TM)";

inline int fpga_selector_v(const device &device) {
  return detail::selectDeviceByPlatform(HARDWARE_PLATFORM_NAME, device);
}

inline int fpga_emulator_selector_v(const device &device) {
  return detail::selectDeviceByPlatform(EMULATION_PLATFORM_NAME, device);
}

inline int fpga_simulator_selector_v(const device &device) {
  static bool IsFirstCall = true;
  if (IsFirstCall) {
    detail::enableFPGASimulator();
    IsFirstCall = false;
  }
  return fpga_selector_v(device);
}

class __SYCL2020_DEPRECATED(
    "Use the callable sycl::ext::intel::fpga_selector_v instead.") fpga_selector
    : public platform_selector {
public:
  fpga_selector() : platform_selector(HARDWARE_PLATFORM_NAME) {}
};

class __SYCL2020_DEPRECATED(
    "Use the callable sycl::ext::intel::fpga_emulator_selector_v instead.")
    fpga_emulator_selector : public platform_selector {
public:
  fpga_emulator_selector() : platform_selector(EMULATION_PLATFORM_NAME) {}
};

class __SYCL2020_DEPRECATED(
    "Use the callable sycl::ext::intel::fpga_simulator_selector_v instead.")
    fpga_simulator_selector : public fpga_selector {
public:
  fpga_simulator_selector() {
    // Tell the runtime to use a simulator device rather than hardware
    detail::enableFPGASimulator();
  }
};

} // namespace ext::intel

} // namespace _V1
} // namespace sycl
