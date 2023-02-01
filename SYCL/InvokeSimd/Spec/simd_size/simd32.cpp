// Test intended to run on PVC only
// TODO: enable on Windows once driver is ready
// REQUIRES: gpu-intel-pvc && linux
// UNSUPPORTED: cuda || hip
//
// TODO: enable when Jira ticket resolved
// XFAIL: gpu
//
// Check that full compilation works:
// RUN: %clangxx -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %s -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out
//
// VISALTO enable run
// RUN: env IGC_VISALTO=63 IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out

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
  passed &= test<4, 32>(q);
  passed &= test<8, 32>(q);
  passed &= test<16, 32>(q);
  passed &= test<32, 32>(q);
  // TODO FIXME: enable cases with ret val size > 32 when Jira ticket resolved

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
