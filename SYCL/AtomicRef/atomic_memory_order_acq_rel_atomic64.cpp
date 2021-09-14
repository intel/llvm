// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// L0, OpenCL, and HIP backends don't currently support
// info::device::atomic_memory_order_capabilities
// UNSUPPORTED: level_zero || opencl || hip

// NOTE: Tests load and store for acquire-release memory ordering with 64-bit
//       values.

#include "atomic_memory_order_acq_rel.h"
#include <iostream>
using namespace sycl;

int main() {
  queue q;

  std::vector<memory_order> supported_memory_orders =
      q.get_device().get_info<info::device::atomic_memory_order_capabilities>();

  if (!is_supported(supported_memory_orders, memory_order::acq_rel) ||
      !q.get_device().has(aspect::atomic64)) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 32;

  // Acquire-release memory order must also support both acquire and release
  // orderings.
  assert(is_supported(supported_memory_orders, memory_order::acquire) &&
         is_supported(supported_memory_orders, memory_order::release));
  acq_rel_test<double>(q, N);

  // Include long tests if they are 64 bits wide
  if constexpr (sizeof(long) == 8) {
    acq_rel_test<long>(q, N);
    acq_rel_test<unsigned long>(q, N);
  }

  // Include long long tests if they are 64 bits wide
  if constexpr (sizeof(long long) == 8) {
    acq_rel_test<long long>(q, N);
    acq_rel_test<unsigned long long>(q, N);
  }

  std::cout << "Test passed." << std::endl;
}
