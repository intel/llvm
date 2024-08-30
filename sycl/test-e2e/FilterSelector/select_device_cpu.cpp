// RUN: %{build} -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR='*:cpu' %{run-unfiltered-devices} %t.out
//
// Checks if only specified device types can be acquired from select_device
// when ONEAPI_DEVICE_SELECTOR is set
// Checks that no device is selected when no device of desired type is
// available.
//
// REQUIRES: opencl,cpu

#include <iostream>

#include <sycl/ext/oneapi/filter_selector.hpp>

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
           "default_selector failed to find cpu device");
  }
  {
    try {
      device d(gpu_selector_v);
      std::cerr << "GPU Device is found: " << std::boolalpha << d.is_gpu()
                << std::endl;
      return -1;
    } catch (...) {
    }
  }
  { device d(cpu_selector_v); }
  {
    try {
      device d(accelerator_selector_v);
      std::cerr << "ACC device is found in error: " << d.is_accelerator()
                << std::endl;
      return -1;
    } catch (...) {
    }
  }

  return 0;
}
