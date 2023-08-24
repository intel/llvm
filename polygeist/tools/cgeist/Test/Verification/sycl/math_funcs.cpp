// Test that `sycl.math.*` operations are constructed ...
// RUN: clang++  -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown-syclmlir \
// RUN:     -O0 -w -emit-mlir -S -o %t.mlir %s
// RUN: FileCheck %s --check-prefix=MLIR -DMLIR_TYPE=f32 < %t.mlir
// RUN: FileCheck %s --check-prefix=MLIR -DMLIR_TYPE=f64 < %t.mlir
// RUN: FileCheck %s --check-prefix=MLIR '-DMLIR_TYPE=!sycl_half' < %t.mlir
// RUN: FileCheck %s --check-prefix=MLIR '-DMLIR_TYPE=!sycl_vec_f32_2_' < %t.mlir
// RUN: FileCheck %s --check-prefix=MLIR '-DMLIR_TYPE=!sycl_vec_f64_2_' < %t.mlir
// RUN: FileCheck %s --check-prefix=MLIR '-DMLIR_TYPE=!sycl_vec_sycl_half_2_' < %t.mlir

// ... and lowered to the corresponding LLVM intrinsics.
// RUN: clang++  -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown-syclmlir \
// RUN:     -O0 -w -emit-llvm -S -o %t.ll %s
// RUN: FileCheck %s --check-prefix=LLVM -DLLVM_TYPE=float -DINTR_TYPE=f32 < %t.ll
// RUN: FileCheck %s --check-prefix=LLVM -DLLVM_TYPE=double -DINTR_TYPE=f64 < %t.ll
// RUN: FileCheck %s --check-prefix=LLVM -DLLVM_TYPE=half -DINTR_TYPE=f16 < %t.ll
// RUN: FileCheck %s --check-prefix=LLVM '-DLLVM_TYPE=<2 x float>' -DINTR_TYPE=v2f32 < %t.ll
// RUN: FileCheck %s --check-prefix=LLVM '-DLLVM_TYPE=<2 x double>' -DINTR_TYPE=v2f64 < %t.ll
// RUN: FileCheck %s --check-prefix=LLVM '-DLLVM_TYPE=<2 x half>' -DINTR_TYPE=v2f16 < %t.ll

// Test that the LLVMIR generated is verifiable.
// RUN: opt -passes=verify -disable-output %t.ll

// Verify that LLVMIR generated is translatable to SPIRV.
// RUN: llvm-as %t.ll && llvm-spirv %t.bc

#include <sycl/sycl.hpp>

template<typename T>
SYCL_EXTERNAL T math_funcs(T a, T b, T c) {
  // MLIR: sycl.math.ceil %{{.*}} : [[MLIR_TYPE]]
  // LLVM: call [[LLVM_TYPE]] @llvm.ceil.[[INTR_TYPE]]
  a += sycl::ceil(a);
  // MLIR: sycl.math.copysign %{{.*}}, %{{.*}} : [[MLIR_TYPE]]
  // LLVM: call [[LLVM_TYPE]] @llvm.copysign.[[INTR_TYPE]]
  a += sycl::copysign(a, b);
  // MLIR: sycl.math.cos %{{.*}} : [[MLIR_TYPE]]
  // LLVM: call [[LLVM_TYPE]] @llvm.cos.[[INTR_TYPE]]
  a += sycl::cos(a);
  // MLIR: sycl.math.exp %{{.*}} : [[MLIR_TYPE]]
  // LLVM: call [[LLVM_TYPE]] @llvm.exp.[[INTR_TYPE]]
  a += sycl::exp(a);
  // MLIR: sycl.math.exp2 %{{.*}} : [[MLIR_TYPE]]
  // LLVM: call [[LLVM_TYPE]] @llvm.exp2.[[INTR_TYPE]]
  a += sycl::exp2(a);
  // MLIR: sycl.math.expm1 %{{.*}} : [[MLIR_TYPE]]
  // LLVM: %[[exp:.*]] = call [[LLVM_TYPE]] @llvm.exp.[[INTR_TYPE]]
  // LLVM-NEXT: fsub [[LLVM_TYPE]] %[[exp]], {{1\.000000e\+00|0xH3C00|<.*>}}
  a += sycl::expm1(a);
  // MLIR: sycl.math.fabs %{{.*}} : [[MLIR_TYPE]]
  // LLVM: call [[LLVM_TYPE]] @llvm.fabs.[[INTR_TYPE]]
  a += sycl::fabs(a);
  // MLIR: sycl.math.floor %{{.*}} : [[MLIR_TYPE]]
  // LLVM: call [[LLVM_TYPE]] @llvm.floor.[[INTR_TYPE]]
  a += sycl::floor(a);
  // MLIR: sycl.math.fma %{{.*}}, %{{.*}}, %{{.*}} : [[MLIR_TYPE]]
  // LLVM: call [[LLVM_TYPE]] @llvm.fma.[[INTR_TYPE]]
  a += sycl::fma(a, b, c);
  // MLIR: sycl.math.log %{{.*}} : [[MLIR_TYPE]]
  // LLVM: call [[LLVM_TYPE]] @llvm.log.[[INTR_TYPE]]
  a += sycl::log(a);
  // MLIR: sycl.math.log10 %{{.*}} : [[MLIR_TYPE]]
  // LLVM: call [[LLVM_TYPE]] @llvm.log10.[[INTR_TYPE]]
  a += sycl::log10(a);
  // MLIR: sycl.math.log2 %{{.*}} : [[MLIR_TYPE]]
  // LLVM: call [[LLVM_TYPE]] @llvm.log2.[[INTR_TYPE]]
  a += sycl::log2(a);
  // MLIR: sycl.math.pow %{{.*}} : [[MLIR_TYPE]]
  // LLVM: call [[LLVM_TYPE]] @llvm.pow.[[INTR_TYPE]]
  a += sycl::pow(a, a);
  // MLIR: sycl.math.round %{{.*}} : [[MLIR_TYPE]]
  // LLVM: call [[LLVM_TYPE]] @llvm.round.[[INTR_TYPE]]
  a += sycl::round(a);
  // MLIR: sycl.math.rsqrt %{{.*}} : [[MLIR_TYPE]]
  // LLVM: %[[sqrt:.*]] = call [[LLVM_TYPE]] @llvm.sqrt.[[INTR_TYPE]]
  // LLVM-NEXT: fdiv [[LLVM_TYPE]] {{1\.000000e\+00|0xH3C00|<.*>}}, %[[sqrt]]
  a += sycl::rsqrt(a);
  // MLIR: sycl.math.sin %{{.*}} : [[MLIR_TYPE]]
  // LLVM: call [[LLVM_TYPE]] @llvm.sin.[[INTR_TYPE]]
  a += sycl::sin(a);
  // MLIR: sycl.math.sqrt %{{.*}} : [[MLIR_TYPE]]
  // LLVM: call [[LLVM_TYPE]] @llvm.sqrt.[[INTR_TYPE]]
  a += sycl::sqrt(a);
  // MLIR: sycl.math.trunc %{{.*}} : [[MLIR_TYPE]]
  // LLVM: call [[LLVM_TYPE]] @llvm.trunc.[[INTR_TYPE]]
  a += sycl::trunc(a);
  
  return a;
}

SYCL_EXTERNAL double test_math_funcs(double a, double b, double c) {
  double res = 0.0;
  res += math_funcs<float>(a, b, c);
  res += math_funcs<double>(a, b, c);
  res += math_funcs<sycl::half>(a, b, c);

  sycl::float2 vfres = math_funcs(
    sycl::float2{a, b}, sycl::float2{c, a}, sycl::float2{b, c});
  sycl::double2 vdres = math_funcs(
    sycl::double2{a, b}, sycl::double2{c, a}, sycl::double2{b, c});
  sycl::half2 vhres = math_funcs(
    sycl::half2{a, b}, sycl::half2{c, a}, sycl::half2{b, c});

  return res + sycl::length(vfres) + sycl::length(vdres) + sycl::length(vhres);
}
