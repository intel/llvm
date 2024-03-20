// REQUIRES: gpu, level_zero, level_zero_dev_kit
// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out 2>&1 | FileCheck %s

#include <iostream>
#include <sycl/sycl.hpp>

// Check for queries of USM capabilities.
// All supported L0 devices have these capabilities currently:
//
// CHECK:  usm_host_allocations: 1
// CHECK:  usm_device_allocations: 1
// CHECK:  usm_shared_allocations: 1
// CHECK:  usm_system_allocations: 0
// CHECK:  usm_atomic_host_allocations: 0
// usm_atomic_shared_allocations is device and driver version dependent.

using namespace sycl;

int main() {
  auto D = device(gpu_selector_v);
  std::cout << "name = " << D.get_info<info::device::name>() << std::endl;

  std::cout << "  usm_host_allocations: " << D.has(aspect::usm_host_allocations)
            << std::endl;
  std::cout << "  usm_device_allocations: "
            << D.has(aspect::usm_device_allocations) << std::endl;
  std::cout << "  usm_shared_allocations: "
            << D.has(aspect::usm_shared_allocations) << std::endl;
  std::cout << "  usm_system_allocations: "
            << D.has(aspect::usm_system_allocations) << std::endl;
  std::cout << "  usm_atomic_host_allocations: "
            << D.has(aspect::usm_atomic_host_allocations) << std::endl;
  std::cout << "  usm_atomic_shared_allocations: "
            << D.has(aspect::usm_atomic_shared_allocations) << std::endl;
  return 0;
}
