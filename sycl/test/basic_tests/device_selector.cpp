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
// Checks if all different device types can be acquired from select_device
// regardless of env var setting SYCL_BE and/or SYCL_DEVICE_TYPE
// Checks that no device is selected when no device of desired type is
// available.

#include <CL/sycl.hpp>

#include <iostream>

using namespace cl::sycl;

class RejectEverything : public device_selector {
public:
  int operator()(const device &Device) const final {
    // Negative value means that a device must not be selected
    return -1;
  }
};

int main() {
  default_selector ds;
  device d = ds.select_device(info::device_type::gpu, backend::level0);
  std::cout << "Level-zero GPU Device is found: " << std::boolalpha << d.is_gpu() << std::endl;
  d = ds.select_device(info::device_type::gpu, backend::opencl);
  std::cout << "OpenCL GPU Device is found: " << std::boolalpha << d.is_gpu() << std::endl;
  d = ds.select_device(info::device_type::cpu);
  std::cout << "CPU device is found: " << d.is_cpu() << std::endl;
  d = ds.select_device(info::device_type::host);
  std::cout << "HOST device is found: " << d.is_host() << std::endl;
  d = ds.select_device(info::device_type::accelerator);
  std::cout << "ACC device is found: " << d.is_accelerator() << std::endl;

  RejectEverything Selector;
  try {
    sycl::device Device(Selector);
  } catch (runtime_error &E) {
    return 0;
  }
  std::cerr << "Error. A device is found." << std::endl;
  return 1;
}
