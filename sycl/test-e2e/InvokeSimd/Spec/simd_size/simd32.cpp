// Test intended to run on PVC only
// REQUIRES: arch-intel_gpu_pvc
//
// Check that full compilation works:
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} %t.out

/* Tests invoke_simd support in the compiler/headers
 * Test checks support for different subgroup sizes in combination with
 * different return value sizes
 *
 * PVC => Register size: 512b
 *
 * This test also runs with all types of VISA link time optimizations enabled.
 */

#include "Inputs/common.hpp"

int main(void) {
  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  bool passed = true;

  // simd_size 32
  passed &= test<32, 32>(q);

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
