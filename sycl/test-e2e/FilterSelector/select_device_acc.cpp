// RUN: %{build} -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR='*:fpga' %{run-unfiltered-devices} %t.out
//
// Checks if only specified device types can be acquired from select_device
// when ONEAPI_DEVICE_SELECTOR is set
// Checks that no device is selected when no device of desired type is
// available.
//
// REQUIRES: opencl,accelerator

#include <iostream>

#include <sycl/detail/core.hpp>

using namespace sycl;
using namespace std;

int main() {
  const char *envVal = std::getenv("ONEAPI_DEVICE_SELECTOR");
  std::string forcedPIs;
  if (envVal) {
    forcedPIs = envVal;
  }
  {
    device d(default_selector_v);
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("OpenCL") != string::npos &&
           "default selector failed to find acc device");
  }
  {
    try {
      device d(gpu_selector_v);
      std::cerr << "GPU Device is found in error: " << std::boolalpha
                << d.is_gpu() << std::endl;
      return -1;
    } catch (...) {
    }
  }
  {
    try {
      device d(cpu_selector_v);
      std::cerr << "CPU Device is found in error: " << std::boolalpha
                << d.is_cpu() << std::endl;
      return -1;
    } catch (...) {
    }
  }
  {
    device d(accelerator_selector_v);
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("OpenCL") != string::npos &&
           "accelerator_selector failed to find acc device");
  }

  return 0;
}
