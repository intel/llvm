// https://github.com/intel/llvm/issues/14244
// sycl-ls --verbose shows the "ext_intel_free_memory" aspect only if
// ZES_ENABLE_SYSMAN=1 is set, so this test is missed if it requires
// aspect-ext_intel_free_memory. Since gen9 and get12 don't support this query,
// so requiring DG2. There may be more devices in our CI supporting this aspect.
// REQUIRES: gpu-intel-dg2
// REQUIRES: level_zero, level_zero_dev_kit
// UNSUPPORTED: gpu-intel-gen9, gpu-intel-gen12
// The query of free memory is not supported on integrated devices
//
// RUN: %{build} %level_zero_options -o %t.out
// RUN: env ZES_ENABLE_SYSMAN=1 %{run} %t.out 2>&1 | FileCheck %s
//
// The test is to check that the free device memory is reported by Level Zero
// backend
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
