// The test verifies support of bf8/hf8 <-> half conversions

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
  simd<half, 16> FP16 = 0;
  simd<bf8, 16> BF8 = FP16;
  // CHECK: call <16 x i8> @llvm.genx.qf.cvt.v16i8.v16f16(<16 x half> {{[^)]+}})
  simd<half, 16> FP16_conv = BF8;
  // CHECK: call <16 x half> @llvm.genx.qf.cvt.v16f16.v16i8(<16 x i8> {{[^)]+}})

  simd<hf8, 16> HF8 = FP16;
  // CHECK: call <16 x i8> @llvm.genx.hf8.cvt.v16i8.v16f16(<16 x half> {{[^)]+}})
  simd<half, 16> FP16_conv1 = HF8;
  // CHECK: call <16 x half> @llvm.genx.hf8.cvt.v16f16.v16i8(<16 x i8> {{[^)]+}})
}
