// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
// RUN: env SYCL_BE=PI_OPENCL %t.out
// RUN: env SYCL_BE=PI_LEVEL0 %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

// This test checks if CPU device is not filtered out by setting SYCL_BE.

#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
  auto devices = device::get_devices(info::device_type::cpu);
  assert(devices.size() == 1 && "CPU device should be available even when SYCL_BE is set");
  return 0;
}
