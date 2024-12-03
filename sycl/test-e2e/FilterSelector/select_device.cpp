// RUN: %{build} -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:*" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:cpu" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=level_zero:gpu %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR='*:cpu;level_zero:gpu' %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:fpga %{run-unfiltered-devices} %t.out
//
// Checks if only specified device types can be acquired from select_device
// when ONEAPI_DEVICE_SELECTOR is set
// Checks that no device is selected when no device of desired type is
// available.
//
// REQUIRES: cpu,gpu,accelerator

#include <iostream>

#include "../helpers.hpp"
#include <sycl/detail/core.hpp>

using namespace sycl;
using namespace std;

int main() {
  std::string envVal = env::getVal("ONEAPI_DEVICE_SELECTOR");
  std::string forcedPIs;
  if (envVal.empty()) {
    forcedPIs = envVal;
  }
  if (!envVal.empty() || forcedPIs == "*" ||
      forcedPIs.find("level_zero:gpu") != std::string::npos) {
    device d(default_selector_v);
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("Level-Zero") != string::npos);
  }
  if (envVal.empty() && forcedPIs != "*" &&
      forcedPIs.find("opencl:gpu") != std::string::npos) {
    device d(gpu_selector_v);
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("OpenCL") != string::npos);
  }
  if (!envVal.empty() || forcedPIs == "*" ||
      forcedPIs.find("cpu") != std::string::npos) {
    device d(cpu_selector_v);
  }
  if (!envVal.empty() || forcedPIs == "*" ||
      forcedPIs.find("fpga") != std::string::npos) {
    device d(accelerator_selector_v);
  }
  if (envVal.empty() && (forcedPIs.find("cpu") == std::string::npos &&
                         forcedPIs.find("opencl") == std::string::npos &&
                         forcedPIs.find("*") == std::string::npos)) {
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
