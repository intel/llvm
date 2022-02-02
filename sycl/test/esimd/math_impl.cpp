// RUN: %clangxx -fsycl -c -fsycl-device-only -Xclang -emit-llvm %s -o - | FileCheck %s

// This test checks the codegen for the following ESIMD APIs:
// sin, cos, exp, log.

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <CL/sycl/builtins_esimd.hpp>

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental;
using namespace sycl::ext::intel::experimental::esimd;

// Math sin,cos,log,exp functions are translated into scalar __spirv_ocl_ calls
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL simd<float, 16> sycl_math(simd<float, 16> x) {
  simd<float, 16> v = 0;
  //CHECK: call spir_func noundef float @_Z15__spirv_ocl_cosf
  v = sycl::cos(x);
  //CHECK: call spir_func noundef float @_Z15__spirv_ocl_sinf
  v = sycl::sin(v);
  //CHECK: call spir_func noundef float @_Z15__spirv_ocl_logf
  v = sycl::log(v);
  //CHECK: call spir_func noundef float @_Z15__spirv_ocl_expf
  v = sycl::exp(v);
  return v;
}

// Math sin,cos,log2,exp2 functions from esimd namespace are translated
// into vector __esimd_ calls, which later translate into GenX intrinsics.
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL simd<float, 16>
esimd_math(simd<float, 16> x) {
  simd<float, 16> v = 0;
  //CHECK: call spir_func noundef <16 x float> @_Z11__esimd_cos
  v = esimd::cos(x);
  //CHECK: call spir_func noundef <16 x float> @_Z11__esimd_sin
  v = esimd::sin(v);
  //CHECK: call spir_func noundef <16 x float> @_Z11__esimd_log
  v = esimd::log2(v);
  //CHECK: call spir_func noundef <16 x float> @_Z11__esimd_exp
  v = esimd::exp2(v);
  return v;
}

// Math log,exp functions from esimd namespace are emulated with
// __esimd_ log/exp calls, which later translate into GenX intrinsics.
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL simd<float, 16>
esimd_math_emu(simd<float, 16> x) {
  simd<float, 16> v = 0;
  //CHECK: call spir_func noundef <16 x float> @_Z11__esimd_log
  v = esimd::log(x);
  //CHECK: call spir_func noundef <16 x float> @_Z11__esimd_exp
  v = esimd::exp(v);
  return v;
}
