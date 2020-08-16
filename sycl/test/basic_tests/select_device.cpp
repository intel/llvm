// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out
// RUN: env SYCL_DEVICE_TRIPLES="*" %t.out
// RUN: env SYCL_DEVICE_TRIPLES=gpu:level_zero %t.out
// RUN: env SYCL_DEVICE_TRIPLES=cpu,acc %t.out
// RUN: env SYCL_DEVICE_TRIPLES="*:opencl" %t.out
// RUN: env SYCL_DEVICE_TRIPLES="*:opencl,gpu:level_zero" %t.out
// RUN: env SYCL_DEVICE_TRIPLES=acc:opencl:0 %t.out
//
// Checks that only designated plugins are loaded when SYCL_DEVICE_TRIPLES is
// set. Checks that all different device types can be acquired from
// select_device()
// UNSUPPORTED: windows

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

int main() {
  const char *pis = std::getenv("SYCL_DEVICE_TRIPLES");
  std::string forcedPIs;
  if (pis) {
    forcedPIs = pis;
  }

  default_selector ds;
  if (!pis || forcedPIs == "*" ||
      forcedPIs.find("gpu:level_zero") != std::string::npos) {
    device d = ds.select_device(info::device_type::gpu, backend::level_zero);
    std::cout << "Level-zero GPU Device is found: " << std::boolalpha
              << d.is_gpu() << std::endl;
  }
  if (!pis || forcedPIs == "*" ||
      forcedPIs.find("opencl") != std::string::npos) {
    device d = ds.select_device(info::device_type::gpu, backend::opencl);
    std::cout << "OpenCL GPU Device is found: " << std::boolalpha << d.is_gpu()
              << std::endl;
  }
  if (!pis || forcedPIs == "*" ||
      forcedPIs.find("opencl") != std::string::npos ||
      forcedPIs.find("cpu") != std::string::npos) {
    device d = ds.select_device(info::device_type::cpu);
    std::cout << "CPU device is found: " << d.is_cpu() << std::endl;
  }
  // HOST device is always available.
  {
    device d = ds.select_device(info::device_type::host);
    std::cout << "HOST device is found: " << d.is_host() << std::endl;
  }
  if (!pis || forcedPIs == "*" ||
      forcedPIs.find("opencl") != std::string::npos ||
      forcedPIs.find("acc") != std::string::npos) {
    device d = ds.select_device(info::device_type::accelerator);
    std::cout << "ACC device is found: " << d.is_accelerator() << std::endl;
  }
  /*
  // Enable the following tests after https://github.com/intel/llvm/pull/2239
  // is merged.
  // If SYCL_DEVICE_TRIPLES is set with level_zero,
  // CPU device should not be found by get_devices(info::device_type::cpu)
  // but GPU should be found by select_device(info::device_type::gpu).
  if (pis && forcedPIs.find("level_zero") != std::string::npos &&
      forcedPIs.find("opencl") == std::string::npos &&
      forcedPIs.find("cpu") == std::string::npos &&
      forcedPIs != "*") {
    auto devices = device::get_devices(info::device_type::cpu);
    for (const device& d : devices) {
      assert(!d.is_cpu() &&
        "Error: CPU device is found when SYCL_DEVICE_TRIPLES sets level_zero");
    }
    device d = ds.select_device(info::device_type::gpu, backend::level_zero);
    assert(d.is_gpu() && "Error: GPU device is not found by select_device.");
  }

  // CPU device should not be loaded if SYCL_DEVICE_TRIPLES does not
  // include 'opencl' string.
  if (pis && forcedPIs.find("opencl") == std::string::npos &&
      forcedPIs.find("cpu") == std::string::npos &&
      forcedPIs.find("*") == std::string::npos) {
    device d = ds.select_device(info::device_type::cpu);
    assert(!d.is_cpu() && "Error: CPU device is found when opencl is not
  loaded");
  }
  */
  return 0;
}
