// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER="*" %t.out
// RUN: env SYCL_DEVICE_FILTER=cpu %t.out
// RUN: env SYCL_DEVICE_FILTER=level_zero:gpu %t.out
// RUN: env SYCL_DEVICE_FILTER=opencl:gpu %t.out
// RUN: env SYCL_DEVICE_FILTER=cpu,level_zero:gpu %t.out
// RUN: env SYCL_DEVICE_FILTER=opencl:acc %t.out
//
// Checks if only specified device types can be acquired from select_device
// when SYCL_DEVICE_FILTER is set
// Checks that no device is selected when no device of desired type is
// available.
//
// REQUIRES: cpu,gpu,accelerator

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
  if (!envVal || forcedPIs == "*" ||
      forcedPIs.find("level_zero:gpu") != std::string::npos) {
    default_selector ds;
    device d = ds.select_device();
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("Level-Zero") != string::npos);
    std::cout << "Level-zero GPU Device is found: " << std::boolalpha
              << d.is_gpu() << std::endl;
  }
  if (envVal && forcedPIs != "*" &&
      forcedPIs.find("opencl:gpu") != std::string::npos) {
    gpu_selector gs;
    device d = gs.select_device();
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("OpenCL") != string::npos);
    std::cout << "OpenCL GPU Device is found: " << std::boolalpha << d.is_gpu()
              << std::endl;
  }
  if (!envVal || forcedPIs == "*" ||
      forcedPIs.find("cpu") != std::string::npos) {
    cpu_selector cs;
    device d = cs.select_device();
    std::cout << "CPU device is found: " << d.is_cpu() << std::endl;
  }
  if (!envVal || forcedPIs == "*" ||
      forcedPIs.find("acc") != std::string::npos) {
    accelerator_selector as;
    device d = as.select_device();
    std::cout << "ACC device is found: " << d.is_accelerator() << std::endl;
  }
  if (envVal && (forcedPIs.find("cpu") == std::string::npos &&
                 forcedPIs.find("opencl") == std::string::npos &&
                 forcedPIs.find("*") == std::string::npos)) {
    try {
      cpu_selector cs;
      device d = cs.select_device();
    } catch (...) {
      std::cout << "Expectedly, CPU device is not found." << std::endl;
      return 0; // expected
    }
    std::cerr << "Error: CPU device is found" << std::endl;
    return -1;
  }

  return 0;
}
