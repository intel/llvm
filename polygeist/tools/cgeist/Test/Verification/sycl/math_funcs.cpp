// Test that `sycl.math.*` operations are constructed ...
// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown-syclmlir \
// RUN:     -O0 -w -emit-mlir -S -o - %s | FileCheck %s --check-prefixes=BOTH,MLIR

// ... and lowered to the corresponding LLVM intrinsics.
// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown-syclmlir \
// RUN:     -O0 -w -emit-llvm -S -o %t.ll %s && FileCheck %s --check-prefixes=BOTH,LLVM < %t.ll

// Test that the LLVMIR generated is verifiable.
// RUN: opt -passes=verify -disable-output %t.ll

// Verify that LLVMIR generated is translatable to SPIRV.
// RUN: llvm-as %t.ll && llvm-spirv %t.bc

#include <sycl/sycl.hpp>

// BOTH-LABEL: math_funcs_float
SYCL_EXTERNAL float math_funcs_float(float a, float b, float c) {
  // MLIR: sycl.math.ceil
  // LLVM: call float @llvm.ceil.f32
  a += sycl::ceil(a);
  // MLIR: sycl.math.copysign
  // LLVM: call float @llvm.copysign.f32
  a += sycl::copysign(a, b);
  // MLIR: sycl.math.cos
  // LLVM: call float @llvm.cos.f32
  a += sycl::cos(a);
  // MLIR: sycl.math.exp
  // LLVM: call float @llvm.exp.f32
  a += sycl::exp(a);
  // MLIR: sycl.math.exp2
  // LLVM: call float @llvm.exp2.f32
  a += sycl::exp2(a);
  // MLIR: sycl.math.expm1
  // LLVM: %[[exp:.*]] = call float @llvm.exp.f32
  // LLVM-NEXT: fsub float %[[exp]], 1.000000e+00
  a += sycl::expm1(a);
  // MLIR: sycl.math.fabs
  // LLVM: call float @llvm.fabs.f32
  a += sycl::fabs(a);
  // MLIR: sycl.math.floor
  // LLVM: call float @llvm.floor.f32
  a += sycl::floor(a);
  // MLIR: sycl.math.fma
  // LLVM: call float @llvm.fma.f32
  a += sycl::fma(a, b, c);
  // MLIR: sycl.math.log
  // LLVM: call float @llvm.log.f32
  a += sycl::log(a);
  // MLIR: sycl.math.log10
  // LLVM: call float @llvm.log10.f32
  a += sycl::log10(a);
  // MLIR: sycl.math.log2
  // LLVM: call float @llvm.log2.f32
  a += sycl::log2(a);
  // MLIR: sycl.math.pow
  // LLVM: call float @llvm.pow.f32
  a += sycl::pow(a, a);
  // MLIR: sycl.math.round
  // LLVM: call float @llvm.round.f32
  a += sycl::round(a);
  // MLIR: sycl.math.rsqrt
  // LLVM: %[[sqrt:.*]] = call float @llvm.sqrt.f32
  // LLVM-NEXT: fdiv float 1.000000e+00, %[[sqrt]]
  a += sycl::rsqrt(a);
  // MLIR: sycl.math.sin
  // LLVM: call float @llvm.sin.f32
  a += sycl::sin(a);
  // MLIR: sycl.math.sqrt
  // LLVM: call float @llvm.sqrt.f32
  a += sycl::sqrt(a);
  // MLIR: sycl.math.trunc
  // LLVM: call float @llvm.trunc.f32
  a += sycl::trunc(a);
  
  return a;
}
