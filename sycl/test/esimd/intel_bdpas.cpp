// The test verifies BDPAS API with FP8, bf16, hf16 and FP4 types.

// Checks that lowerESIMD pass builds proper vc-intrinsics
// RUN: %clangxx -O0 -fsycl -c -fsycl-device-only -Xclang -emit-llvm %s -o %t
// -O0 lowering, requires `-force-disable-esimd-opt` to disable all
// optimizations.
// RUN: sycl-post-link -split-esimd -lower-esimd -O0 -force-disable-esimd-opt -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_0.esimd.ll

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::esimd;
using bf8 = sycl::ext::intel::experimental::esimd::bf8;
using hf8 = sycl::ext::intel::experimental::esimd::hf8;
using half = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using s1e2m1 = sycl::ext::intel::experimental::esimd::fp4_S1E2M1;

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void test() {
  constexpr int M = 8;
  constexpr int N = 16;

  simd<uint8_t, M> AScale;
  simd<uint8_t, N> BScale;

  simd<uint8_t, 2 * M> AScaleFP4;
  simd<uint8_t, 2 * N> BScaleFP4;

  simd<float, M * N> C;

  simd<half, M * 16> AHF16;
  simd<half, N * 16> BHF16;

  simd<bf16, M * 16> ABF16;
  simd<bf16, N * 16> BBF16;

  simd<bf8, M * 32> ABF8;
  simd<bf8, N * 32> BBF8;

  simd<hf8, M * 32> AHF8;
  simd<hf8, N * 32> BHF8;

  simd<s1e2m1, M * 32> AS1E2M1;
  simd<s1e2m1, N * 32> BS1E2M1;

  // CHECK: call <128 x float> @llvm.genx.bdpas.v128f32.v128f32.v128i32.v64i32.v16i8.v8i8(<128 x float> {{[^,]+}}, <128 x i32> {{[^,]+}}, <64 x i32> {{[^,]+}}, <16 x i8> {{[^,]+}}, <8 x i8> {{[^,]+}}, i32 10, i32 10, i32 8, i32 8)
  C = __ESIMD_XMX_NS::bdpas<float>(C, BHF16, AHF16, BScale, AScale);
  // CHECK: call <128 x float> @llvm.genx.bdpas.v128f32.v128f32.v128i32.v64i32.v16i8.v8i8(<128 x float> {{[^,]+}}, <128 x i32> {{[^,]+}}, <64 x i32> {{[^,]+}}, <16 x i8> {{[^,]+}}, <8 x i8> {{[^,]+}}, i32 9, i32 9, i32 8, i32 8)
  C = __ESIMD_XMX_NS::bdpas<float>(C, BBF16, ABF16, BScale, AScale);

  // CHECK: call <128 x float> @llvm.genx.bdpas.v128f32.v128f32.v128i32.v64i32.v16i8.v8i8(<128 x float> {{[^,]+}}, <128 x i32> {{[^,]+}}, <64 x i32> {{[^,]+}}, <16 x i8> {{[^,]+}}, <8 x i8> {{[^,]+}}, i32 11, i32 11, i32 8, i32 8)
  C = __ESIMD_XMX_NS::bdpas<float>(C, BBF8, ABF8, BScale, AScale);
  // CHECK: call <128 x float> @llvm.genx.bdpas.v128f32.v128f32.v128i32.v64i32.v16i8.v8i8(<128 x float> {{[^,]+}}, <128 x i32> {{[^,]+}}, <64 x i32> {{[^,]+}}, <16 x i8> {{[^,]+}}, <8 x i8> {{[^,]+}}, i32 11, i32 14, i32 8, i32 8)
  C = __ESIMD_XMX_NS::bdpas<float>(C, BBF8, AHF8, BScale, AScale);
  // CHECK: call <128 x float> @llvm.genx.bdpas.v128f32.v128f32.v128i32.v64i32.v16i8.v8i8(<128 x float> {{[^,]+}}, <128 x i32> {{[^,]+}}, <64 x i32> {{[^,]+}}, <16 x i8> {{[^,]+}}, <8 x i8> {{[^,]+}}, i32 14, i32 11, i32 8, i32 8)
  C = __ESIMD_XMX_NS::bdpas<float>(C, BHF8, ABF8, BScale, AScale);
  // CHECK: call <128 x float> @llvm.genx.bdpas.v128f32.v128f32.v128i32.v64i32.v16i8.v8i8(<128 x float> {{[^,]+}}, <128 x i32> {{[^,]+}}, <64 x i32> {{[^,]+}}, <16 x i8> {{[^,]+}}, <8 x i8> {{[^,]+}}, i32 14, i32 14, i32 8, i32 8)
  C = __ESIMD_XMX_NS::bdpas<float>(C, BHF8, AHF8, BScale, AScale);

  // CHECK: call <128 x float> @llvm.genx.bdpas.v128f32.v128f32.v128i32.v64i32.v32i8.v16i8(<128 x float> {{[^,]+}}, <128 x i32> {{[^,]+}}, <64 x i32> {{[^,]+}}, <32 x i8> {{[^,]+}}, <16 x i8> {{[^,]+}}, i32 15, i32 15, i32 8, i32 8)
  C = __ESIMD_XMX_NS::bdpas<float>(C, BS1E2M1, AS1E2M1, BScaleFP4, AScaleFP4);
}
