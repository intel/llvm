// UNSUPPORTED: cpu
// #2252 Disable until all variants of built-ins are available in OpenCL CPU
// runtime for every supported ISA

// UNSUPPORTED: cuda || rocm

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test verifies the correct work of SPIR-V 1.3 exclusive_scan() and
// inclusive_scan() algoriths used with the MUL operation.

#include "scan.hpp"

int main() {
  queue Queue;
  if (!core_sg_supported(Queue.get_device()) ||
      !Queue.get_device().has_extension("cl_khr_fp64")) {
    std::cout << "Skipping test\n";
    return 0;
  }
  check<class MulDouble, double>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
