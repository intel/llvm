// RUN: %{build} -o %t.out

// clang-format off
// RUN-IF: cpu, env ONEAPI_DEVICE_SELECTOR="*:cpu" %{run-unfiltered-devices} %t.out
// RUN-IF: level_zero && gpu, env ONEAPI_DEVICE_SELECTOR="level_zero:gpu" %{run-unfiltered-devices} %t.out
// RUN-IF: opencl && gpu, env ONEAPI_DEVICE_SELECTOR="opencl:gpu" %{run-unfiltered-devices} %t.out
// RUN-IF: cpu && level_zero && gpu, env ONEAPI_DEVICE_SELECTOR="*:cpu;level_zero:gpu" %{run-unfiltered-devices} %t.out
// clang-format on

//
// Checks if only specified device types can be acquired from select_device
// when ONEAPI_DEVICE_SELECTOR is set
// Checks that no device is selected when no device of desired type is
// available.
//

#include <iostream>

#include "../helpers.hpp"
#include <sycl/detail/core.hpp>
#include <sycl/platform.hpp>

using namespace sycl;
using namespace std;

int main() {
  std::string forcedDevice = env::getVal("ONEAPI_DEVICE_SELECTOR");
  if (forcedDevice == "*:*" ||
      forcedDevice.find("level_zero:gpu") != std::string::npos) {
    device d(default_selector_v);
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("Level-Zero") != string::npos);
  }
  if (forcedDevice != "*:*" &&
      forcedDevice.find("opencl:gpu") != std::string::npos) {
    device d(gpu_selector_v);
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("OpenCL") != string::npos);
  }
  if (forcedDevice == "*:*" || forcedDevice.find("cpu") != std::string::npos) {
    device d(cpu_selector_v);
  }
  if (forcedDevice.find("cpu") == std::string::npos &&
      forcedDevice.find("opencl") == std::string::npos &&
      forcedDevice.find("*") == std::string::npos) {
    try {
      device d(cpu_selector_v);
    } catch (...) {
      return 0; // expected
    }
    std::cerr << "Error: CPU device is found, even though it shouldn't be"
              << std::endl;
    return -1;
  }

  return 0;
}