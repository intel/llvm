// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out
// RUN: env SYCL_DEVICE_TRIPLE=cpu %t.out
// RUN: env SYCL_DEVICE_TRIPLE=gpu:level0 %t.out
// RUN: env SYCL_DEVICE_TRIPLE=gpu:opencl %t.out
// RUN: env SYCL_DEVICE_TRIPLE=cpu,gpu:level0 %t.out
//
// Checks if only specified device types can be acquired from select_device
// when SYCL_DEVICE_TRIPLE is set
// Checks that no device is selected when no device of desired type is
// available.

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

int main() {
  const char *envVal = std::getenv("SYCL_DEVICE_TRIPLE");
  std::string forcedPIs;
  if (envVal) {
    std::cout << "SYCL_DEVICE_TRIPLE=" << envVal << std::endl;
    forcedPIs = envVal;
  }
  if (!envVal || forcedPIs.find("gpu:level0") != std::string::npos) {
    default_selector ds;
    device d = ds.select_device();
    std::cout << "Level-zero GPU Device is found: " << std::boolalpha
              << d.is_gpu() << std::endl;
  }
  if (!envVal || forcedPIs.find("gpu:opencl") != std::string::npos) {
    gpu_selector gs;
    device d = gs.select_device();
    std::cout << "OpenCL GPU Device is found: " << std::boolalpha << d.is_gpu()
              << std::endl;
  }
  if (!envVal || forcedPIs.find("cpu") != std::string::npos) {
    cpu_selector cs;
    device d = cs.select_device();
    std::cout << "CPU device is found: " << d.is_cpu() << std::endl;
  }
  // HOST device is always available regardless of SYCL_DEVICE_TRIPLE
  {
    host_selector hs;
    device d = hs.select_device();
    std::cout << "HOST device is found: " << d.is_host() << std::endl;
  }
  if (!envVal || forcedPIs.find("accelerator") != std::string::npos) {
    accelerator_selector as;
    device d = as.select_device();
    std::cout << "ACC device is found: " << d.is_accelerator() << std::endl;
  }
  if (envVal && (forcedPIs.find("cpu") == std::string::npos &&
                 // remove the following condition when SYCL_DEVICE_TRIPLE
                 // filter works in device selectors
                 forcedPIs.find("opencl") == std::string::npos &&
                 forcedPIs.find("*") == std::string::npos)) {
    try {
      cpu_selector cs;
      device d = cs.select_device();
    } catch (...) {
      std::cout << "Expectedly, CPU device is not found." << std::endl;
      return 0; // expected
    }
    std::cout << "Error: CPU device is found" << std::endl;
    return -1;
  }

  return 0;
}
