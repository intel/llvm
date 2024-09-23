// Check that full compilation works:
// RUN: not %{build} -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %s -o %t.out 2>&1 | FileCheck %s
/*
  Test case specification: Test and report errors if invoked ESIMD function
  is recursive.
*/

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/detail/core.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;
constexpr int VL = 16;

ESIMD_INLINE
esimd::simd<float, VL>
ESIMD_CALLEE(esimd::simd<float, VL> a) SYCL_ESIMD_FUNCTION {
  return a.select<1, 1>(0) == 0 ? a : ESIMD_CALLEE(a + 1);
  // CHECK: {{.*}}error: SYCL kernel cannot call a recursive function{{.*}}
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE(simd<float, VL> a)
        SYCL_ESIMD_FUNCTION;

int main(void) {
  // Test should not be executed, clang compile fail expected.
  return 1;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE(simd<float, VL> a)
        SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE(a);
  return res;
}
