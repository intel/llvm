// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out
// RUN: env SYCL_DEVICE_TRIPLES="*" %t.out
// RUN: env SYCL_DEVICE_TRIPLES=cpu %t.out
// RUN: env SYCL_DEVICE_TRIPLES=level_zero:gpu %t.out
// RUN: env SYCL_DEVICE_TRIPLES=opencl:gpu %t.out
// RUN: env SYCL_DEVICE_TRIPLES=cpu,level_zero:gpu %t.out
// RUN: env SYCL_DEVICE_TRIPLES=opencl:acc:0 %t.out
//
// Checks if only specified device types can be acquired from select_device
// when SYCL_DEVICE_TRIPLES is set
// Checks that no device is selected when no device of desired type is
// available.
// UNSUPPORTED: windows

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

int main() {
  const char *envVal = std::getenv("SYCL_DEVICE_TRIPLES");
  std::string forcedPIs;
  if (envVal) {
    std::cout << "SYCL_DEVICE_TRIPLES=" << envVal << std::endl;
    forcedPIs = envVal;
  }
  if (!envVal || forcedPIs == "*" ||
      forcedPIs.find("gpu:level_zero") != std::string::npos) {
    default_selector ds;
    device d = ds.select_device();
    std::cout << "Level-zero GPU Device is found: " << std::boolalpha
              << d.is_gpu() << std::endl;
  }
  if (!envVal || forcedPIs == "*" ||
      forcedPIs.find("gpu:opencl") != std::string::npos) {
    gpu_selector gs;
    device d = gs.select_device();
    std::cout << "OpenCL GPU Device is found: " << std::boolalpha << d.is_gpu()
              << std::endl;
  }
  if (!envVal || forcedPIs == "*" ||
      forcedPIs.find("cpu") != std::string::npos) {
    cpu_selector cs;
    device d = cs.select_device();
    std::cout << "CPU device is found: " << d.is_cpu() << std::endl;
  }
  // HOST device is always available regardless of SYCL_DEVICE_TRIPLES
  {
    host_selector hs;
    device d = hs.select_device();
    std::cout << "HOST device is found: " << d.is_host() << std::endl;
  }
  if (!envVal || forcedPIs == "*" ||
      forcedPIs.find("acc") != std::string::npos) {
    accelerator_selector as;
    device d = as.select_device();
    std::cout << "ACC device is found: " << d.is_accelerator() << std::endl;
  }
  if (envVal && (forcedPIs.find("cpu") == std::string::npos &&
                 // remove the following condition when SYCL_DEVICE_TRIPLES
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
