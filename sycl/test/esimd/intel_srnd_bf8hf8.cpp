// The test verifies support of bf16/half -> bf8/hf8 conversions

// Checks host+device compilation
// RUN: %clangxx -fsycl -fsyntax-only %s

// Checks that lowerESIMD pass builds proper vc-intrinsics
// RUN: %clangxx -O2 -fsycl -c -fsycl-device-only -Xclang -emit-llvm %s -o %t
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

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void bf8_convert() {
  simd<half, 8> H = 0;
  simd<sycl::ext::oneapi::bfloat16, 8> B = 1;
  simd<uint8_t, 8> bias = 5;

  // CHECK: call <8 x i8> @llvm.genx.biased.rounding.bf8.v8i8.v8f16(<8 x half> {{[^)]+}}, <8 x i8> {{[^)]+}})
  simd<bf8, 8> res1 = __ESIMD_ENS::srnd<bf8>(H, bias);

  // CHECK: call <8 x i8> @llvm.genx.biased.rounding.bf8.v8i8.v8i16(<8 x i16> {{[^)]+}}, <8 x i8> {{[^)]+}})
  simd<bf8, 8> res2 = __ESIMD_ENS::srnd<bf8>(B, bias);

  // CHECK: call <8 x i8> @llvm.genx.biased.rounding.hf8.v8i8.v8f16(<8 x half> {{[^)]+}}, <8 x i8> {{[^)]+}})
  simd<hf8, 8> res3 = __ESIMD_ENS::srnd<hf8>(H, bias);

  // CHECK: call <8 x i8> @llvm.genx.biased.rounding.hf8.v8i8.v8i16(<8 x i16> {{[^)]+}}, <8 x i8> {{[^)]+}})
  simd<hf8, 8> res4 = __ESIMD_ENS::srnd<hf8>(B, bias);
}
