// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR='opencl:*' %t.out
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
    cout << "ONEAPI_DEVICE_SELECTOR=" << envVal << std::endl;
    forcedPIs = envVal;
  }

  {
    default_selector ds;
    device d = ds.select_device();
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("OpenCL") != string::npos);
    cout << "OpenCL GPU Device is found: " << boolalpha << d.is_gpu()
         << std::endl;
  }
  {
    gpu_selector gs;
    device d = gs.select_device();
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("OpenCL") != string::npos);
    cout << name << " is found: " << boolalpha << d.is_gpu() << std::endl;
  }
  {
    cpu_selector cs;
    device d = cs.select_device();
    cout << "CPU device is found : " << d.is_cpu() << std::endl;
  }
  {
    accelerator_selector as;
    device d = as.select_device();
    cout << "ACC device is found : " << d.is_accelerator() << std::endl;
  }

  return 0;
}
