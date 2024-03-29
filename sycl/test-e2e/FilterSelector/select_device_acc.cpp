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
    default_selector ds;
    device d = ds.select_device();
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("OpenCL") != string::npos &&
           "default selector failed to find acc device");
  }
  {
    gpu_selector gs;
    try {
      device d = gs.select_device();
      std::cerr << "GPU Device is found in error: " << std::boolalpha
                << d.is_gpu() << std::endl;
      return -1;
    } catch (...) {
    }
  }
  {
    cpu_selector cs;
    try {
      device d = cs.select_device();
      std::cerr << "CPU Device is found in error: " << std::boolalpha
                << d.is_cpu() << std::endl;
      return -1;
    } catch (...) {
    }
  }
  {
    accelerator_selector as;
    device d = as.select_device();
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("OpenCL") != string::npos &&
           "accelerator_selector failed to find acc device");
  }

  return 0;
}
