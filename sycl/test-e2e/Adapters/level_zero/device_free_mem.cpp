// REQUIRES: aspect-ext_intel_free_memory
// REQUIRES: level_zero, level_zero_dev_kit
//
// RUN: %{build} %level_zero_options -o %t.out
// RUN: env ZES_ENABLE_SYSMAN=1 %{run} %t.out 2>&1 | FileCheck %s
// RUN: env ZES_ENABLE_SYSMAN=0 %{run} %t.out 2>&1 | FileCheck %s
//
// The test is to check that the free device memory is reported by Level Zero
// backend both with and without the sysman environment variable.
//
// CHECK: Root-device free memory

#include <iostream>
#include <sycl/detail/core.hpp>
using namespace sycl;

int main() {

  queue Queue;
  auto dev = Queue.get_device();
  std::cout << "Device: " << dev.get_info<info::device::name>() << std::endl;

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

  return 0;
}
