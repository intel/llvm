// RUN: %{build} -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR='opencl:*' %{run-unfiltered-devices} %t.out
//
// Checks if only specified device types can be acquired from select_device
// when ONEAPI_DEVICE_SELECTOR is set
// Checks that no device is selected when no device of desired type is
// available.
//
// REQUIRES: opencl,gpu,cpu,accelerator

#include "../helpers.hpp"
#include <iostream>
#include <sycl/detail/core.hpp>

using namespace sycl;
using namespace std;

int main() {
  std::string envVal = env::getVal("ONEAPI_DEVICE_SELECTOR");
  string forcedPIs;
  if (envVal.empty()) {
    forcedPIs = envVal;
  }

  {
    device d(default_selector_v);
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("OpenCL") != string::npos &&
           "default_selector_v failed to find an opencl device");
  }
  {
    device d(gpu_selector_v);
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("OpenCL") != string::npos &&
           "gpu_selector_v failed to find an opencl device");
  }
  {
    device d(cpu_selector_v);
  }
  {
    device d(accelerator_selector_v);
  }

  return 0;
}
