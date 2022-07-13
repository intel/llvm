// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER=cuda:gpu %t.out
//
// Checks if only specified device types can be acquired from select_device
// when SYCL_DEVICE_FILTER is set.
// Checks that no device is selected when no device of desired type is
// available.
//
// REQUIRES: cuda,gpu

#include <iostream>
#include <sycl/sycl.hpp>

using namespace cl::sycl;
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
    assert(name.find("CUDA") != string::npos);
    cout << "CUDA GPU Device is found: " << boolalpha << d.is_gpu()
         << std::endl;
  }
  {
    gpu_selector gs;
    device d = gs.select_device();
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("CUDA") != string::npos);
    cout << name << " is found: " << boolalpha << d.is_gpu() << std::endl;
  }
  {
    cpu_selector cs;
    try {
      device d = cs.select_device();
      cerr << "CPU device is found in error: " << d.is_cpu() << std::endl;
      return -1;
    } catch (...) {
      cout << "Expectedly, cpu device is not found." << std::endl;
    }
  }
  {
    host_selector hs;
    try {
      device d = hs.select_device();
      cerr << "HOST device is found in error: " << d.is_host() << std::endl;
      return -1;
    } catch (...) {
      cout << "Expectedly, HOST device is not found." << std::endl;
    }
  }
  {
    accelerator_selector as;
    try {
      device d = as.select_device();
      cerr << "ACC device is found in error: " << d.is_accelerator()
           << std::endl;
    } catch (...) {
      cout << "Expectedly, ACC device is not found." << std::endl;
    }
  }

  return 0;
}
