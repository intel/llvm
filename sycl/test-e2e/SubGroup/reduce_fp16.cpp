// REQUIRES: aspect-fp16
// REQUIRES: gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Missing __spirv_GroupFAdd, __spirv_GroupFMin, __spirv_GroupFMax on AMD
// XFAIL: hip_amd

// This test verifies the correct work of the sub-group algorithm reduce().

#include "reduce.hpp"

int main() {
  queue Queue;
  if (!core_sg_supported(Queue.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }
  check<class KernelName_oMg, sycl::half>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
