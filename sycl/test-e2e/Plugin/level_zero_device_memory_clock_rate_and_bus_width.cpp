// REQUIRES: level_zero, level_zero_dev_kit
//
// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out 2>&1 | FileCheck %s
//
// The test is to check that the memory clock rate and bus width is reported be
// Level Zero backend
//
// CHECK: Memory clock rate
// CHECK: Memory bus width

#include <iostream>
#include <sycl/detail/core.hpp>
using namespace sycl;

int main() {

  queue Queue;
  auto dev = Queue.get_device();
  std::cout << "Device: " << dev.get_info<info::device::name>() << std::endl;

  if (dev.has(aspect::ext_intel_memory_clock_rate)) {
    auto MemoryClockRate =
        dev.get_info<ext::intel::info::device::memory_clock_rate>();
    std::cout << "Memory clock rate: " << MemoryClockRate << std::endl;
  } else {
    std::cout << "Query ext_intel_device_info_memory_clock_rate not supported "
                 "by the device"
              << std::endl;
  }

  if (dev.has(aspect::ext_intel_memory_bus_width)) {
    auto MemoryBusWidth =
        dev.get_info<ext::intel::info::device::memory_bus_width>();
    std::cout << "Memory bus width: " << MemoryBusWidth << std::endl;
  } else {
    std::cout << "Query ext_intel_device_info_memory_bus_width not supported "
                 "by the device"
              << std::endl;
  }

  return 0;
}
