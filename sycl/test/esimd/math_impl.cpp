// RUN: %clangxx -fsycl -c -fsycl-device-only -Xclang -emit-llvm %s -o - | FileCheck %s

// This test checks the codegen for the following ESIMD APIs:
// sin, cos, exp, log.

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <CL/sycl/builtins_esimd.hpp>

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

// Math sin,cos,log,exp functions are translated into scalar __spirv_ocl_ calls
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL simd<float, 16> sycl_math(simd<float, 16> x) {
  simd<float, 16> v = 0;
  //CHECK: call spir_func float @_Z15__spirv_ocl_cosf
  v = sycl::cos(x);
  //CHECK: call spir_func float @_Z15__spirv_ocl_sinf
  v = sycl::sin(v);
  //CHECK: call spir_func float @_Z15__spirv_ocl_logf
  v = sycl::log(v);
  //CHECK: call spir_func float @_Z15__spirv_ocl_expf
  v = sycl::exp(v);
  return v;
}

// Math sin,cos,log,exp functions from esimd namespace are translated
// into vector __esimd_ calls, which later translate into GenX intrinsics.
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL simd<float, 16>
esimd_math(simd<float, 16> x) {
  simd<float, 16> v = 0;
  //CHECK: call spir_func <16 x float> @_Z11__esimd_cos
  v = esimd_cos(x);
  //CHECK: call spir_func <16 x float> @_Z11__esimd_sin
  v = esimd_sin(v);
  //CHECK: call spir_func <16 x float> @_Z11__esimd_log
  v = esimd_log(v);
  //CHECK: call spir_func <16 x float> @_Z11__esimd_exp
  v = esimd_exp(v);
  return v;
}
