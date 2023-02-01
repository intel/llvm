// TODO: enable on Windows once driver is ready
// REQUIRES: gpu && linux
// UNSUPPORTED: cuda || hip
//
// Check that full compilation works:
// RUN: not %clangxx -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %s -o %t.out 2>&1 | FileCheck %s
//
// Tests invoke_simd support in the compiler/headers

/* Test case specification: Test and report errors if invoked ESIMD function
 * calls SPMD function.
 *
 * This test case is based on the assumption that ESIMD functions can't call
 * SPMD functions, as there is no mechanism for this described in ESIMD
 * specification.
 *
 * In this test case, ESIMD_CALLEE_doVadd() calls SPMD_CALLEE_doVadd().
 * Currently, the compilation fails, which is good.
 */

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;

// 1024 / 16 = 64: There will be 64 iterations that process 16 elements each
constexpr int Size = 1024;
constexpr int VL = 16;

/* Here are 2 possible interpretations for the SPMD functions. Each function
 * contributes to a vector add operation, but the first one is performed on
 * scalar values and the other is performed on vectors. They both cause
 * compile-time errors, but the errors are different. Currently, the second,
 * vector-based function does not compile, even if it's not called. That is, the
 * definition itself is not allowed (hence it's commented out for now). The
 * scalar-based function definition compiles without error, but a call to it is
 * not allowed by the compiler and gives the following error:
 *
 * "error: SYCL device function cannot be called from an ESIMD context"
 */
SYCL_EXTERNAL float SPMD_CALLEE_doVadd_scalar(float a, float b) {
  return a + b;
}

/* This SPMD function is called by an ESIMD function and should not compile. */
SYCL_EXTERNAL simd<float, VL> SPMD_CALLEE_doVadd_vector(simd<float, VL> va,
                                                        simd<float, VL> vb) {
  // SYCL kernel cannot call an undefined function without SYCL_EXTERNAL
  // attribute emulate '+' on simd operands
  simd<float, VL> vc;
  for (int i = 0; i < VL; ++i) {
    vc[i] = va[i] + vb[i];
  }
  return vc;
}

/* Value of this flag doesn't really matter as both 'if constexpr' branches are
 * compiled and both must fail the same way.
 */
constexpr bool use_vector = true; // false;

__attribute__((always_inline)) esimd::simd<float, VL>
ESIMD_CALLEE_doVadd(esimd::simd<float, VL> va,
                    esimd::simd<float, VL> vb) SYCL_ESIMD_FUNCTION {
  if constexpr (use_vector) {
    return SPMD_CALLEE_doVadd_vector(va, vb);
    // CHECK: {{.*}}error: SYCL device function cannot be called from an ESIMD context{{.*}}
  }

  esimd::simd<float, VL> vc;
  for (int i = 0; i < VL; ++i) {
    float a = va[i];
    float b = vb[i];
    // This should cause an error.
    vc[i] = SPMD_CALLEE_doVadd_scalar(a, b);
    // CHECK: {{.*}}error: SYCL device function cannot be called from an ESIMD context{{.*}}
  }
  return vc;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE_doVadd(
        simd<float, VL> va, simd<float, VL> vb) SYCL_ESIMD_FUNCTION;

int main(void) {
  // Test should not be executed, clang compile fail expected.
  return 1;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE_doVadd(
        simd<float, VL> va, simd<float, VL> vb) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE_doVadd(va, vb);
  return res;
}
