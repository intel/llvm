// REQUIRES: level_zero, level_zero_dev_kit
// UNSUPPORTED: gpu-intel-gen9, gpu-intel-gen12
// The query of free memory is not supported on integrated devices
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env ZES_ENABLE_SYSMAN=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
//
// The test is to check that the free device memory is reported by Level Zero
// backend
//
// CHECK: Root-device free memory

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {

  queue Queue;
  auto dev = Queue.get_device();
  std::cout << "Device: " << dev.get_info<info::device::name>() << std::endl;

  if (!dev.is_host() && dev.has(aspect::ext_intel_free_memory)) {
    auto TotalMemory = dev.get_info<sycl::info::device::global_mem_size>();
    auto FreeMemory = dev.get_info<ext::intel::info::device::free_memory>();
    std::cout << "Root-device total memory: " << TotalMemory << std::endl;
    std::cout << "Root-device free  memory: " << FreeMemory << std::endl;
    assert(TotalMemory >= FreeMemory);

    try { // guard for when no partitioning is supported

      auto sub_devices = dev.create_sub_devices<
          info::partition_property::partition_by_affinity_domain>(
          info::partition_affinity_domain::next_partitionable);

      int I = 0;
      for (auto &sub_device : sub_devices) {
        ++I;
        auto SubDeviceTotalMemory =
            sub_device.get_info<sycl::info::device::global_mem_size>();
        auto SubDeviceFreeMemory =
            sub_device.get_info<ext::intel::info::device::free_memory>();
        std::cout << I << " sub-device total memory: " << SubDeviceTotalMemory
                  << std::endl;
        std::cout << I << " sub-device free  memory: " << SubDeviceFreeMemory
                  << std::endl;
        assert(SubDeviceFreeMemory <= FreeMemory);
        assert(SubDeviceTotalMemory >= SubDeviceFreeMemory);
        assert(SubDeviceTotalMemory <= TotalMemory);
      }

    } catch (...) {
    }

  } else {
    std::cout
        << "Query ext_intel_device_info_free_memory not supported by the device"
        << std::endl;
  }

  return 0;
}
