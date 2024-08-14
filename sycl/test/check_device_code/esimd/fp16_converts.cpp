// The test verifies support of bfloat16 <-> float conversions

// Checks host+device compilation
// RUN: %clangxx -fsycl -fsyntax-only %s

// Checks that lowerESIMD pass builds proper vc-intrinsics
// RUN: %clangxx -O2 -fsycl -c -fsycl-device-only -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -properties -split-esimd -lower-esimd -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::esimd;

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void bf16_vector();
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void bf16_scalar();

using bfloat16 = sycl::ext::oneapi::bfloat16;

class EsimdFunctor {
public:
  void operator()() __attribute__((sycl_explicit_simd)) {
    bf16_vector();
    bf16_scalar();
  }
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void bf16_vector() {
  simd<float, 8> F32 = 0;
  simd<bfloat16, 8> BF16 = F32;
  // CHECK: call <8 x half> @llvm.genx.bf.cvt.v8f16.v8f32(<8 x float> {{[^)]+}})
  simd<float, 8> F32_conv = BF16;
  // CHECK: call <8 x float> @llvm.genx.bf.cvt.v8f32.v8f16(<8 x half> {{[^)]+}})
}

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void bf16_scalar() {
  // Note that this is the compilation test only. It checks that IR is correct.
  // The actual support in GPU RT is on the way though.
  float F32_scalar = 1;
  bfloat16 BF16_scalar = F32_scalar;
  // CHECK: call i16 @__spirv_ConvertFToBF16INTEL(float {{[^)]+}})
  float F32_scalar_conv = BF16_scalar;
  // CHECK: call float @__spirv_ConvertBF16ToFINTEL(i16 {{[^)]+}})
}
