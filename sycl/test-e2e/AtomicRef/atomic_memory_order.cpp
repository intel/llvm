// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test checks whether the minimum required memory order capabilities are
// supported in both context and device queries. Specifically the "relaxed"
// memory order capability, which is used in other tests.

#include "atomic_memory_order.h"
#include <cassert>
#include <iostream>
using namespace sycl;

int main() {
  queue q;

  // Context
  std::vector<memory_order> supported_context_memory_orders =
      q.get_context()
          .get_info<info::context::atomic_memory_order_capabilities>();

  assert(is_supported(supported_context_memory_orders, memory_order::relaxed));

  // Device
  std::vector<memory_order> supported_device_memory_orders =
      q.get_device().get_info<info::device::atomic_memory_order_capabilities>();

  assert(is_supported(supported_device_memory_orders, memory_order::relaxed));

  std::cout << "Test passed." << std::endl;
}
