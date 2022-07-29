// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER=opencl,host %t.out
//
// Checks if only specified device types can be acquired from select_device
// when SYCL_DEVICE_FILTER is set
// Checks that no device is selected when no device of desired type is
// available.
//
// REQUIRES: opencl,gpu,cpu,accelerator

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace std;

int main() {
  const char *envVal = getenv("SYCL_DEVICE_FILTER");
  string forcedPIs;
  if (envVal) {
    cout << "SYCL_DEVICE_FILTER=" << envVal << std::endl;
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
    host_selector hs;
    device d = hs.select_device();
    cout << "HOST device is found: " << d.is_host() << std::endl;
  }
  {
    accelerator_selector as;
    device d = as.select_device();
    cout << "ACC device is found : " << d.is_accelerator() << std::endl;
  }

  return 0;
}
