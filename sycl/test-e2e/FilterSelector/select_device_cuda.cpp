// RUN: %{build} -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=cuda:gpu %{run-unfiltered-devices} %t.out
//
// Checks if only specified device types can be acquired from select_device
// when ONEAPI_DEVICE_SELECTOR is set.
// Checks that no device is selected when no device of desired type is
// available.
//
// REQUIRES: cuda,gpu

#include <iostream>

#include "../helpers.hpp"
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
    assert(name.find("CUDA") != string::npos);
  }
  {
    device d(gpu_selector_v);
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("CUDA") != string::npos);
  }
  {
    try {
      device d(cpu_selector_v);
      cerr << "CPU device is found in error: " << d.is_cpu() << std::endl;
      return -1;
    } catch (...) {
    }
  }
  {
    try {
      device d(accelerator_selector_v);
      cerr << "ACC device is found in error: " << d.is_accelerator()
           << std::endl;
    } catch (...) {
    }
  }

  return 0;
}
