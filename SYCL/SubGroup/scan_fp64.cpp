// UNSUPPORTED: cpu
// #2252 Disable until all variants of built-ins are available in OpenCL CPU
// runtime for every supported ISA
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test verifies the correct work of the sub-group algorithms
// exclusive_scan() and inclusive_scan().

#include "scan.hpp"

int main() {
  queue Queue;
  if (!core_sg_supported(Queue.get_device()) ||
      !Queue.get_device().has(sycl::aspect::fp64)) {
    std::cout << "Skipping test\n";
    return 0;
  }
  check<class KernelName_cYZflKkIXS, double>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
