// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// L0, OpenCL, and ROCm backends don't currently support
// info::device::atomic_memory_order_capabilities
// UNSUPPORTED: level_zero || opencl || rocm

// NOTE: Tests load and store for sequentially consistent memory ordering.

#include "atomic_memory_order_seq_cst.h"
#include <iostream>
using namespace sycl;

int main() {
  queue q;

  std::vector<memory_order> supported_memory_orders =
      q.get_device().get_info<info::device::atomic_memory_order_capabilities>();

  if (!is_supported(supported_memory_orders, memory_order::seq_cst)) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 32;

  seq_cst_test<int>(q, N);
  seq_cst_test<unsigned int>(q, N);
  seq_cst_test<float>(q, N);

  // Include long tests if they are 32 bits wide
  if constexpr (sizeof(long) == 4) {
    seq_cst_test<long>(q, N);
    seq_cst_test<unsigned long>(q, N);
  }

  std::cout << "Test passed." << std::endl;
}
