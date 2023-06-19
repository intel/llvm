// This is the second module for split_module/SPMD_module.cpp test

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;
constexpr int VL = 16;

esimd::simd<float, VL> ESIMD_CALLEE(float *A, esimd::simd<float, VL> b,
                                    int i) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> a;
  a.copy_from(A + i);
  return a + b;
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall SIMD_CALLEE(float *A, simd<float, VL> b,
                                          int i) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> res = ESIMD_CALLEE(A, b, i);
  return res;
}
