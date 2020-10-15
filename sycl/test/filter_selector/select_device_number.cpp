// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RU: env SYCL_DEVICE_FILTER=2 %t.out
//
// Checks if only specified device types can be acquired from select_device
// when SYCL_DEVICE_FILTER is set
// Checks that no device is selected when no device of desired type is
// available.
//
// REQUIRES: opencl,cpu,gpu,accelerator,host

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
    assert(name.find("OpenCL") != string::npos);
    std::cout << "CPU Device is found: " << std::boolalpha << d.is_cpu()
              << std::endl;
  }
  {
    cpu_selector cs;
    try {
      device d = cs.select_device();
      std::cout << "CPU Device is found: " << std::boolalpha
                << d.is_cpu() << std::endl;
    } catch (...) {
      std::cout << "Unexpectedly, CPU device is not found." << std::endl;
      return -1;
    }
  }
  // HOST device is always available regardless of SYCL_DEVICE_FILTER
  {
    host_selector hs;
    device d = hs.select_device();
    std::cout << "HOST device is found: " << d.is_host() << std::endl;
  }

  return 0;
}
