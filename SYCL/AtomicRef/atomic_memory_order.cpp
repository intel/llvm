// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// L0, OpenCL, and HIP backends don't currently support
// info::device::atomic_memory_order_capabilities
// UNSUPPORTED: level_zero || opencl || hip

// NOTE: General tests for atomic memory order capabilities.

#include "atomic_memory_order.h"
#include <cassert>
#include <iostream>
using namespace sycl;

int main() {
  queue q;

  std::vector<memory_order> supported_memory_orders =
      q.get_device().get_info<info::device::atomic_memory_order_capabilities>();

  // Relaxed memory order must be supported. This ordering is used in other
  // tests.
  assert(is_supported(supported_memory_orders, memory_order::relaxed));

  std::cout << "Test passed." << std::endl;
}
