// REQUIRES: level_zero, level_zero_dev_kit
// UNSUPPORTED: gpu-intel-gen9, gpu-intel-gen12
// The query of free memory is not supported on integrated devices
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env ZES_ENABLE_SYSMAN=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
//
// The test is to check that the free device memory is reported be Level Zero
// backend
//
// CHECK: Free device memory

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {

  queue Queue;
  auto dev = Queue.get_device();
  std::cout << "Device: " << dev.get_info<info::device::name>() << std::endl;

  if (!dev.is_host() && dev.has(aspect::ext_intel_free_memory)) {
    auto FreeMemory = dev.get_info<ext::intel::info::device::free_memory>();
    std::cout << "Free device memory: " << FreeMemory << std::endl;
  } else {
    std::cout
        << "Query ext_intel_device_info_free_memory not supported by the device"
        << std::endl;
  }

  return 0;
}
