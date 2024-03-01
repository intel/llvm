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
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace std;

int main() {
  const char *envVal = std::getenv("ONEAPI_DEVICE_SELECTOR");
  std::string forcedPIs;
  if (envVal) {
    forcedPIs = envVal;
  }
  if (!envVal || forcedPIs == "*" ||
      forcedPIs.find("level_zero:gpu") != std::string::npos) {
    default_selector ds;
    device d = ds.select_device();
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("Level-Zero") != string::npos);
  }
  if (envVal && forcedPIs != "*" &&
      forcedPIs.find("opencl:gpu") != std::string::npos) {
    gpu_selector gs;
    device d = gs.select_device();
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("OpenCL") != string::npos);
  }
  if (!envVal || forcedPIs == "*" ||
      forcedPIs.find("cpu") != std::string::npos) {
    cpu_selector cs;
    device d = cs.select_device();
  }
  if (!envVal || forcedPIs == "*" ||
      forcedPIs.find("fpga") != std::string::npos) {
    accelerator_selector as;
    device d = as.select_device();
  }
  if (envVal && (forcedPIs.find("cpu") == std::string::npos &&
                 forcedPIs.find("opencl") == std::string::npos &&
                 forcedPIs.find("*") == std::string::npos)) {
    try {
      cpu_selector cs;
      device d = cs.select_device();
    } catch (...) {
      return 0; // expected
    }
    std::cerr << "Error: CPU device is found, even though it shouldn't be"
              << std::endl;
    return -1;
  }

  return 0;
}
