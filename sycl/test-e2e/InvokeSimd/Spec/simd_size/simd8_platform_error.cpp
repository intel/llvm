// Test intended to run on PVC only
// TODO: enable on Windows once driver is ready
// REQUIRES: gpu-intel-pvc && linux
//
// UNSUPPORTED: cuda || hip
//
// Check that full compilation works:
// RUN: %clangxx -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %s -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER not %t.out 2>&1 | FileCheck %s

/* Tests invoke_simd support in the compiler/headers
 * The test checks error message running simd8 on PVC
 */

#include "Inputs/common.hpp"

int main(void) {
  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  bool passed = true;

  // VL = 8
  passed &= test<8, 8>(q);
  // CHECK: {{.*}}error: Kernel compiled with required subgroup size 8, which is unsupported on this platform{{.*}}

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
