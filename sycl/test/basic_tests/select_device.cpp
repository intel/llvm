// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out
// UN: env SYCL_DEVICE_TRIPLE=gpu:level0 %t.out
// UN: env SYCL_DEVICE_TRIPLE=cpu,acc %t.out
// UN: env SYCL_DEVICE_TRIPLE=*:opencl %t.out
// UN: env SYCL_DEVICE_TRIPLE=*:opencl,gpu:level0 %t.out
//
// Checks that only designated plugins are loaded when SYCL_DEVICE_TRIPLE is
// set. Checks that all different device types can be acquired from
// select_device()
// UNSUPPORTED: windows

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

int main() {
  const char *pis = std::getenv("SYCL_DEVICE_TRIPLE");
  std::string forcedPIs;
  if (pis) {
    forcedPIs = pis;
  }

  default_selector ds;
  if (!pis || forcedPIs.find("gpu:level0") != std::string::npos) {
    device d = ds.select_device(info::device_type::gpu, backend::level_zero);
    std::cout << "Level-zero GPU Device is found: " << std::boolalpha
              << d.is_gpu() << std::endl;
  }
  if (!pis || forcedPIs.find("opencl") != std::string::npos) {
    device d = ds.select_device(info::device_type::gpu, backend::opencl);
    std::cout << "OpenCL GPU Device is found: " << std::boolalpha << d.is_gpu()
              << std::endl;
  }
  if (!pis || forcedPIs.find("opencl") != std::string::npos ||
      forcedPIs.find("cpu") != std::string::npos) {
    device d = ds.select_device(info::device_type::cpu);
    std::cout << "CPU device is found: " << d.is_cpu() << std::endl;
  }
  // HOST device is always available.
  {
    device d = ds.select_device(info::device_type::host);
    std::cout << "HOST device is found: " << d.is_host() << std::endl;
  }
  if (!pis || forcedPIs.find("opencl") != std::string::npos ||
      forcedPIs.find("acc") != std::string::npos) {
    device d = ds.select_device(info::device_type::accelerator);
    std::cout << "ACC device is found: " << d.is_accelerator() << std::endl;
  }
  // If SYCL_DEVICE_TRIPLE is set with level0,
  // GPU device should not be found by get_devices(info::device_type::gpu)
  // but found by select_device(info::device_type::gpu).
  if (pis && forcedPIs.find("level0") != std::string::npos &&
      forcedPIs.find("opencl") == std::string::npos &&
      forcedPIs.find("cpu") == std::string::npos &&
      forcedPIs.find("*") == std::string::npos) {
    auto devices = device::get_devices(info::device_type::gpu);
    assert(
        devices.size() == 0 &&
        "Error: CPU device is found when SYCL_DEVICE_TRIPLE contains level0");
    device d = ds.select_device(info::device_type::gpu, backend::level_zero);
    assert(d.is_gpu() && "Error: GPU device is not found by select_device.");
  }
  // CPU device should not be loaded if SYCL_DEVICE_TRIPLE does not
  // include 'opencl' string.
  if (pis && forcedPIs.find("opencl") == std::string::npos &&
      forcedPIs.find("cpu") == std::string::npos &&
      forcedPIs.find("*") == std::string::npos) {
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
