// Test not intended to run on PVC
// UNSUPPORTED: arch-intel_gpu_pvc
//
// TODO: enable when Jira ticket resolved
// XFAIL: gpu
//
// Check that full compilation works:
// RUN: %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %{run} not %t.out | FileCheck %s

/* Tests invoke_simd support in the compiler/headers
 * The test checks error message running simd32 on devices other than PVC
 */

#include "Inputs/common.hpp"

int main(void) {
  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  bool passed = true;

  // VL = 32
  passed &= test<32, 32>(q);
  // CHECK: {{.*}}error: Kernel compiled with required subgroup size 32, which is unsupported on this platform{{.*}}

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
