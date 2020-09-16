// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RU: env SYCL_DEVICE_FILTER=cpu %t.out
//
// Checks if only specified device types can be acquired from select_device
// when SYCL_DEVICE_FILTER is set
// Checks that no device is selected when no device of desired type is
// available.
//
// REQUIRES: opencl,host,cpu

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;
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
    assert(name.find("OpneCL") != string::npos);
    std::cout << "CPU Device is found: " << std::boolalpha
              << d.is_cpu() << std::endl;
  }
  {
    gpu_selector gs;
    try {
      device d = gs.select_device();
      std::cerr << "GPU Device is found: " << std::boolalpha << d.is_gpu()
		<< std::endl;
      return -1;
    } catch (...) {
      std::cout << "Expectedly, GPU device is not found." << std::endl;
    }
  }
  {
    cpu_selector cs;
    device d = cs.select_device();
    std::cout << "CPU device is found: " << d.is_cpu() << std::endl;
  }
  // HOST device is always available regardless of SYCL_DEVICE_FILTER
  {
    host_selector hs;
    device d = hs.select_device();
    std::cout << "HOST device is found: " << d.is_host() << std::endl;
  }
  {
    accelerator_selector as;
    try {
      device d = as.select_device();
      std::cerr << "ACC device is found in error: " << d.is_accelerator() << std::endl;
      return -1;
    } catch(...) {
      std::cout << "Expectedly, ACC device is not found." << std::endl;
    }
  }

  return 0;
}
