// RUN: %{build} -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR='opencl:*' %{run-unfiltered-devices} %t.out
//
// Checks if only specified device types can be acquired from select_device
// when ONEAPI_DEVICE_SELECTOR is set
// Checks that no device is selected when no device of desired type is
// available.
//
// REQUIRES: opencl,gpu,cpu,accelerator

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace std;

int main() {
  const char *envVal = getenv("ONEAPI_DEVICE_SELECTOR");
  string forcedPIs;
  if (envVal) {
    forcedPIs = envVal;
  }

  {
    default_selector ds;
    device d = ds.select_device();
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("OpenCL") != string::npos &&
           "default_selector failed to find an opencl device");
  }
  {
    gpu_selector gs;
    device d = gs.select_device();
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("OpenCL") != string::npos &&
           "gpu_selector failed to find an opencl device");
  }
  {
    cpu_selector cs;
    device d = cs.select_device();
  }
  {
    accelerator_selector as;
    device d = as.select_device();
  }

  return 0;
}
