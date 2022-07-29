// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER=acc %t.out
//
// Checks if only specified device types can be acquired from select_device
// when SYCL_DEVICE_FILTER is set
// Checks that no device is selected when no device of desired type is
// available.
//
// REQUIRES: opencl,accelerator

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace std;

int main() {
  const char *envVal = std::getenv("SYCL_DEVICE_FILTER");
  std::string forcedPIs;
  if (envVal) {
    std::cout << "SYCL_DEVICE_FILTER=" << envVal << std::endl;
    forcedPIs = envVal;
  }
  {
    default_selector ds;
    device d = ds.select_device();
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("OpenCL") != string::npos);
    std::cout << "ACC Device is found: " << std::boolalpha << d.is_accelerator()
              << std::endl;
  }
  {
    gpu_selector gs;
    try {
      device d = gs.select_device();
      std::cerr << "GPU Device is found in error: " << std::boolalpha
                << d.is_gpu() << std::endl;
      return -1;
    } catch (...) {
      std::cout << "Expectedly, GPU device is not found." << std::endl;
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
      std::cout << "Expectedly, CPU device is not found." << std::endl;
    }
  }
  /*
  // TODO: enable this test after SYCL_DEVICE_FILTER is merged.
  {
    host_selector hs;
    try {
      device d = hs.select_device();
      std::cerr << "HOST Device is found in error: " << std::boolalpha
                << d.is_cpu() << std::endl;
      return -1;
    } catch (...) {
      std::cout << "Expectedly, HOST device is not found." << std::endl;
    }
  }
  */
  {
    accelerator_selector as;
    device d = as.select_device();
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("OpenCL") != string::npos);
    std::cout << "ACC device is found: " << d.is_accelerator() << std::endl;
  }

  return 0;
}
