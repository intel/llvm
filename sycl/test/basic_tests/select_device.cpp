// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_BE=%sycl_be %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: env SYCL_DEVICE_TYPE=CPU %t.out
// RUN: env SYCL_DEVICE_TYPE=GPU %t.out
// RUN: env SYCL_DEVICE_TYPE=ACC %t.out
// RUN: env SYCL_DEVICE_TYPE=GPU SYCL_BE=%sycl_be %t.out
// RUN: env SYCL_DEVICE_TYPE=CPU SYCL_BE=%sycl_be %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST SYCL_BE=%sycl_be %t.out
//
// Checks that only designated plugins are loaded when SYCL_FORCE_PI is set.
// Checks that all different device types can be acquired from select_device()
// regardless of env var setting SYCL_BE and/or SYCL_DEVICE_TYPE.

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

int main() {
  const char *pis = std::getenv("SYCL_FORCE_PI");
  const char *be = std::getenv("SYCL_BE");
  std::string forcedPIs;
  std::string forcedBE;
  if (pis) {
    forcedPIs = pis;
  }
  if (be) {
    forcedBE = be;
  }
  default_selector ds;
  if (!pis || forcedPIs.find("level0") != std::string::npos) {
    device d = ds.select_device(info::device_type::gpu, backend::level0);
    std::cout << "Level-zero GPU Device is found: " << std::boolalpha << d.is_gpu() << std::endl;
  }
  if (!pis || forcedPIs.find("opencl") != std::string::npos) {
    device d = ds.select_device(info::device_type::gpu, backend::opencl);
    std::cout << "OpenCL GPU Device is found: " << std::boolalpha << d.is_gpu() << std::endl;
  }
  if (!pis || forcedPIs.find("opencl") != std::string::npos) {
    device d = ds.select_device(info::device_type::cpu);
    std::cout << "CPU device is found: " << d.is_cpu() << std::endl;
  }
  // HOST device is always available.
  {
    device d = ds.select_device(info::device_type::host);
    std::cout << "HOST device is found: " << d.is_host() << std::endl;
  }
  if (!pis || forcedPIs.find("opencl") != std::string::npos) {
    device d = ds.select_device(info::device_type::accelerator);
    std::cout << "ACC device is found: " << d.is_accelerator() << std::endl;
  }
  // If SYCL_FORCE_BE is not set and SYCL_BE is set to PI_LEVEL0,
  // CPU device should not be found by get_devices() but found by select_device().
  if (!pis && forcedBE == "PI_LEVEL0") {
      auto devices = device::get_devices(info::device_type::cpu);
      assert(devices.size() == 0 && "Error: CPU device is found when SYCL_BE=PI_LEVEL0");
      device d = ds.select_device(info::device_type::cpu);
      assert(d.is_cpu() && "Error: CPU device is not found by select_device.");
  }  
  // CPU device should not be loaded if SYCL_FORCE_BE does not include 'opencl' string.
  if (pis && forcedPIs.find("opencl") == std::string::npos) {
    try {
      device d = ds.select_device(info::device_type::cpu);
    } catch (...) {
      std::cout << "Expectedly, CPU device is not found." << std::endl;
      return 0;
    }
    std::cout << "Error: CPU device is found" << std::endl;
    return -1;
  }
    device d = ds.select_device(info::device_type::gpu, backend::opencl);
  
  return 0;
}
