// TODO: enable on Windows once driver is ready
// REQUIRES: gpu && linux
// UNSUPPORTED: cuda || hip
//
// Check that full compilation works:
// RUN: not %clangxx -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %s -o %t.out 2>&1 | FileCheck %s
/*
  Test case specification: Test and report errors if sycl::marray argument
  passed to invoked ESIMD function.
*/

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;
constexpr int VL = 16;

ESIMD_INLINE
esimd::simd<float, VL>
ESIMD_CALLEE(float *A, esimd::simd<float, VL> b, int i,
             sycl::marray<float, VL> m) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> a;
  a.copy_from(A + i);
  return a + b + m[i % VL];
  // CHECK: {{.*}}error: function 'sycl::{{.*}}marray<{{.*}}' is not supported in ESIMD context{{.*}}
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE(float *A, simd<float, VL> b, int i,
                                          sycl::marray<float, VL> m)
        SYCL_ESIMD_FUNCTION;

int main(void) {
  // Test should not be executed, clang compile fail expected.
  return 1;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE(float *A, simd<float, VL> b, int i,
                                          sycl::marray<float, VL> m)
        SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE(A, b, i, m);
  return res;
}
