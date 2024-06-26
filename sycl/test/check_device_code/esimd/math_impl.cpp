// RUN: %clangxx -fsycl -c -fsycl-device-only -Xclang -emit-llvm %s -o - | FileCheck %s

// This test checks the codegen for the following ESIMD APIs:
// sin, cos, exp, log.

#include <sycl/builtins_esimd.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel;
using namespace sycl::ext::intel::esimd;
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
  //CHECK:  call spir_func noundef <16 x float> @_Z22__spirv_ocl_native_cos{{[^\(]*}}
  v = esimd::cos(x);
  //CHECK: call spir_func noundef <16 x float> @_Z22__spirv_ocl_native_sin{{[^\(]*}}
  v = esimd::sin(v);
  //CHECK: call spir_func noundef <16 x float> @_Z23__spirv_ocl_native_log2{{[^\(]*}}
  v = esimd::log2(v);
  //CHECK: call spir_func noundef <16 x float> @_Z23__spirv_ocl_native_exp2{{[^\(]*}}
  v = esimd::exp2(v);
  return v;
}

// Math log,exp functions from esimd namespace are emulated with
// __esimd_ log/exp calls, which later translate into GenX intrinsics.
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL simd<float, 16>
esimd_math_emu(simd<float, 16> x) {
  simd<float, 16> v = 0;
  //CHECK: call spir_func noundef <16 x float> @_Z23__spirv_ocl_native_log2{{[^\(]*}}
  v = esimd::log(x);
  //CHECK: call spir_func noundef <16 x float> @_Z23__spirv_ocl_native_exp2{{[^\(]*}}
  v = esimd::exp(v);
  return v;
}

// Logical BNF function from esimd namespace is translated into __esimd_ calls,
// which later translate into GenX intrinsics.
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL simd<int, 16>
esimd_bfn(simd<int, 16> x, simd<int, 16> y, simd<int, 16> z) {
  simd<int, 16> v =
      esimd::bfn<~esimd::bfn_t::x & ~esimd::bfn_t::y & ~esimd::bfn_t::z>(x, y,
                                                                         z);
  //CHECK: call spir_func noundef <16 x i32> @_Z11__esimd_bfn
  return v;
}
