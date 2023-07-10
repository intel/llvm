// RUN: sycl-mlir-opt %s | sycl-mlir-opt | FileCheck %s
// RUN: sycl-mlir-opt %s --mlir-print-op-generic | sycl-mlir-opt | FileCheck %s

// CHECK-LABEL: test_math_ops_float
func.func @test_math_ops_float(%arg0 : f32, %arg1 : f32, %arg2 : f32) {
  // CHECK: %{{.*}} = sycl.math.ceil %arg0 : f32
  %c0 = sycl.math.ceil %arg0 : f32
  // CHECK: %{{.*}} = sycl.math.copysign %arg0, %arg1 : f32
  %c1 = sycl.math.copysign %arg0, %arg1 : f32
  // CHECK: %{{.*}} = sycl.math.cos %arg0 : f32
  %c2 = sycl.math.cos %arg0 : f32
  // CHECK: %{{.*}} = sycl.math.exp %arg0 : f32
  %e2 = sycl.math.exp %arg0 : f32
  // CHECK: %{{.*}} = sycl.math.exp2 %arg0 : f32
  %e0 = sycl.math.exp2 %arg0 : f32
  // CHECK: %{{.*}} = sycl.math.expm1 %arg0 : f32
  %e1 = sycl.math.expm1 %arg0 : f32
  // CHECK: %{{.*}} = sycl.math.fabs %arg0 : f32
  %f0 = sycl.math.fabs %arg0 : f32
  // CHECK: %{{.*}} = sycl.math.floor %arg0 : f32
  %f1 = sycl.math.floor %arg0 : f32
  // CHECK: %{{.*}} = sycl.math.fma %arg0, %arg1, %arg2 : f32
  %f2 = sycl.math.fma %arg0, %arg1, %arg2 : f32
  // CHECK: %{{.*}} = sycl.math.log %arg0 : f32
  %l2 = sycl.math.log %arg0 : f32
  // CHECK: %{{.*}} = sycl.math.log10 %arg0 : f32
  %l0 = sycl.math.log10 %arg0 : f32
  // CHECK: %{{.*}} = sycl.math.log2 %arg0 : f32
  %l1 = sycl.math.log2 %arg0 : f32
  // CHECK: %{{.*}} = sycl.math.pow %arg0, %arg1 : f32
  %p0 = sycl.math.pow %arg0, %arg1 : f32
  // CHECK: %{{.*}} = sycl.math.round %arg0 : f32
  %r0 = sycl.math.round %arg0 : f32
  // CHECK: %{{.*}} = sycl.math.rsqrt %arg0 : f32
  %r1 = sycl.math.rsqrt %arg0 : f32
  // CHECK: %{{.*}} = sycl.math.sin %arg0 : f32
  %s0 = sycl.math.sin %arg0 : f32
  // CHECK: %{{.*}} = sycl.math.sqrt %arg0 : f32
  %s1 = sycl.math.sqrt %arg0 : f32
  // CHECK: %{{.*}} = sycl.math.trunc %arg0 : f32
  %t0 = sycl.math.trunc %arg0 : f32

  return
}

// CHECK-LABEL: test_math_ops_double
func.func @test_math_ops_double(%arg0 : f64, %arg1 : f64, %arg2 : f64) {
  // CHECK: %{{.*}} = sycl.math.ceil %arg0 : f64
  %c0 = sycl.math.ceil %arg0 : f64
  // CHECK: %{{.*}} = sycl.math.copysign %arg0, %arg1 : f64
  %c1 = sycl.math.copysign %arg0, %arg1 : f64
  // CHECK: %{{.*}} = sycl.math.cos %arg0 : f64
  %c2 = sycl.math.cos %arg0 : f64
  // CHECK: %{{.*}} = sycl.math.exp %arg0 : f64
  %e2 = sycl.math.exp %arg0 : f64
  // CHECK: %{{.*}} = sycl.math.exp2 %arg0 : f64
  %e0 = sycl.math.exp2 %arg0 : f64
  // CHECK: %{{.*}} = sycl.math.expm1 %arg0 : f64
  %e1 = sycl.math.expm1 %arg0 : f64
  // CHECK: %{{.*}} = sycl.math.fabs %arg0 : f64
  %f0 = sycl.math.fabs %arg0 : f64
  // CHECK: %{{.*}} = sycl.math.floor %arg0 : f64
  %f1 = sycl.math.floor %arg0 : f64
  // CHECK: %{{.*}} = sycl.math.fma %arg0, %arg1, %arg2 : f64
  %f2 = sycl.math.fma %arg0, %arg1, %arg2 : f64
  // CHECK: %{{.*}} = sycl.math.log %arg0 : f64
  %l2 = sycl.math.log %arg0 : f64
  // CHECK: %{{.*}} = sycl.math.log10 %arg0 : f64
  %l0 = sycl.math.log10 %arg0 : f64
  // CHECK: %{{.*}} = sycl.math.log2 %arg0 : f64
  %l1 = sycl.math.log2 %arg0 : f64
  // CHECK: %{{.*}} = sycl.math.pow %arg0, %arg1 : f64
  %p0 = sycl.math.pow %arg0, %arg1 : f64
  // CHECK: %{{.*}} = sycl.math.round %arg0 : f64
  %r0 = sycl.math.round %arg0 : f64
  // CHECK: %{{.*}} = sycl.math.rsqrt %arg0 : f64
  %r1 = sycl.math.rsqrt %arg0 : f64
  // CHECK: %{{.*}} = sycl.math.sin %arg0 : f64
  %s0 = sycl.math.sin %arg0 : f64
  // CHECK: %{{.*}} = sycl.math.sqrt %arg0 : f64
  %s1 = sycl.math.sqrt %arg0 : f64
  // CHECK: %{{.*}} = sycl.math.trunc %arg0 : f64
  %t0 = sycl.math.trunc %arg0 : f64

  return
}

!sycl_half = !sycl.half<(f16)>

// CHECK-LABEL: test_math_ops_half
func.func @test_math_ops_half(%arg0 : !sycl_half, %arg1 : !sycl_half, %arg2 : !sycl_half) {
  // CHECK: %{{.*}} = sycl.math.ceil %arg0 : !sycl_half
  %c0 = sycl.math.ceil %arg0 : !sycl_half
  // CHECK: %{{.*}} = sycl.math.copysign %arg0, %arg1 : !sycl_half
  %c1 = sycl.math.copysign %arg0, %arg1 : !sycl_half
  // CHECK: %{{.*}} = sycl.math.cos %arg0 : !sycl_half
  %c2 = sycl.math.cos %arg0 : !sycl_half
  // CHECK: %{{.*}} = sycl.math.exp %arg0 : !sycl_half
  %e2 = sycl.math.exp %arg0 : !sycl_half
  // CHECK: %{{.*}} = sycl.math.exp2 %arg0 : !sycl_half
  %e0 = sycl.math.exp2 %arg0 : !sycl_half
  // CHECK: %{{.*}} = sycl.math.expm1 %arg0 : !sycl_half
  %e1 = sycl.math.expm1 %arg0 : !sycl_half
  // CHECK: %{{.*}} = sycl.math.fabs %arg0 : !sycl_half
  %f0 = sycl.math.fabs %arg0 : !sycl_half
  // CHECK: %{{.*}} = sycl.math.floor %arg0 : !sycl_half
  %f1 = sycl.math.floor %arg0 : !sycl_half
  // CHECK: %{{.*}} = sycl.math.fma %arg0, %arg1, %arg2 : !sycl_half
  %f2 = sycl.math.fma %arg0, %arg1, %arg2 : !sycl_half
  // CHECK: %{{.*}} = sycl.math.log %arg0 : !sycl_half
  %l2 = sycl.math.log %arg0 : !sycl_half
  // CHECK: %{{.*}} = sycl.math.log10 %arg0 : !sycl_half
  %l0 = sycl.math.log10 %arg0 : !sycl_half
  // CHECK: %{{.*}} = sycl.math.log2 %arg0 : !sycl_half
  %l1 = sycl.math.log2 %arg0 : !sycl_half
  // CHECK: %{{.*}} = sycl.math.pow %arg0, %arg1 : !sycl_half
  %p0 = sycl.math.pow %arg0, %arg1 : !sycl_half
  // CHECK: %{{.*}} = sycl.math.round %arg0 : !sycl_half
  %r0 = sycl.math.round %arg0 : !sycl_half
  // CHECK: %{{.*}} = sycl.math.rsqrt %arg0 : !sycl_half
  %r1 = sycl.math.rsqrt %arg0 : !sycl_half
  // CHECK: %{{.*}} = sycl.math.sin %arg0 : !sycl_half
  %s0 = sycl.math.sin %arg0 : !sycl_half
  // CHECK: %{{.*}} = sycl.math.sqrt %arg0 : !sycl_half
  %s1 = sycl.math.sqrt %arg0 : !sycl_half
  // CHECK: %{{.*}} = sycl.math.trunc %arg0 : !sycl_half
  %t0 = sycl.math.trunc %arg0 : !sycl_half

  return
}

!sycl_vec_f32_4_ = !sycl.vec<[f32, 4], (vector<4xf32>)>

// CHECK-LABEL: test_math_ops_vector_of_float
func.func @test_math_ops_vector_of_float(%arg0 : !sycl_vec_f32_4_, %arg1 : !sycl_vec_f32_4_, %arg2 : !sycl_vec_f32_4_) {
  // CHECK: %{{.*}} = sycl.math.ceil %arg0 : !sycl_vec_f32_4_
  %c0 = sycl.math.ceil %arg0 : !sycl_vec_f32_4_
  // CHECK: %{{.*}} = sycl.math.copysign %arg0, %arg1 : !sycl_vec_f32_4_
  %c1 = sycl.math.copysign %arg0, %arg1 : !sycl_vec_f32_4_
  // CHECK: %{{.*}} = sycl.math.cos %arg0 : !sycl_vec_f32_4_
  %c2 = sycl.math.cos %arg0 : !sycl_vec_f32_4_
  // CHECK: %{{.*}} = sycl.math.exp %arg0 : !sycl_vec_f32_4_
  %e2 = sycl.math.exp %arg0 : !sycl_vec_f32_4_
  // CHECK: %{{.*}} = sycl.math.exp2 %arg0 : !sycl_vec_f32_4_
  %e0 = sycl.math.exp2 %arg0 : !sycl_vec_f32_4_
  // CHECK: %{{.*}} = sycl.math.expm1 %arg0 : !sycl_vec_f32_4_
  %e1 = sycl.math.expm1 %arg0 : !sycl_vec_f32_4_
  // CHECK: %{{.*}} = sycl.math.fabs %arg0 : !sycl_vec_f32_4_
  %f0 = sycl.math.fabs %arg0 : !sycl_vec_f32_4_
  // CHECK: %{{.*}} = sycl.math.floor %arg0 : !sycl_vec_f32_4_
  %f1 = sycl.math.floor %arg0 : !sycl_vec_f32_4_
  // CHECK: %{{.*}} = sycl.math.fma %arg0, %arg1, %arg2 : !sycl_vec_f32_4_
  %f2 = sycl.math.fma %arg0, %arg1, %arg2 : !sycl_vec_f32_4_
  // CHECK: %{{.*}} = sycl.math.log %arg0 : !sycl_vec_f32_4_
  %l2 = sycl.math.log %arg0 : !sycl_vec_f32_4_
  // CHECK: %{{.*}} = sycl.math.log10 %arg0 : !sycl_vec_f32_4_
  %l0 = sycl.math.log10 %arg0 : !sycl_vec_f32_4_
  // CHECK: %{{.*}} = sycl.math.log2 %arg0 : !sycl_vec_f32_4_
  %l1 = sycl.math.log2 %arg0 : !sycl_vec_f32_4_
  // CHECK: %{{.*}} = sycl.math.pow %arg0, %arg1 : !sycl_vec_f32_4_
  %p0 = sycl.math.pow %arg0, %arg1 : !sycl_vec_f32_4_
  // CHECK: %{{.*}} = sycl.math.round %arg0 : !sycl_vec_f32_4_
  %r0 = sycl.math.round %arg0 : !sycl_vec_f32_4_
  // CHECK: %{{.*}} = sycl.math.rsqrt %arg0 : !sycl_vec_f32_4_
  %r1 = sycl.math.rsqrt %arg0 : !sycl_vec_f32_4_
  // CHECK: %{{.*}} = sycl.math.sin %arg0 : !sycl_vec_f32_4_
  %s0 = sycl.math.sin %arg0 : !sycl_vec_f32_4_
  // CHECK: %{{.*}} = sycl.math.sqrt %arg0 : !sycl_vec_f32_4_
  %s1 = sycl.math.sqrt %arg0 : !sycl_vec_f32_4_
  // CHECK: %{{.*}} = sycl.math.trunc %arg0 : !sycl_vec_f32_4_
  %t0 = sycl.math.trunc %arg0 : !sycl_vec_f32_4_

  return
}

!sycl_vec_f64_8_ = !sycl.vec<[f64, 8], (vector<8xf64>)>

// CHECK-LABEL: test_math_ops_vector_of_double
func.func @test_math_ops_vector_of_double(%arg0 : !sycl_vec_f64_8_, %arg1 : !sycl_vec_f64_8_, %arg2 : !sycl_vec_f64_8_) {
  // CHECK: %{{.*}} = sycl.math.ceil %arg0 : !sycl_vec_f64_8_
  %c0 = sycl.math.ceil %arg0 : !sycl_vec_f64_8_
  // CHECK: %{{.*}} = sycl.math.copysign %arg0, %arg1 : !sycl_vec_f64_8_
  %c1 = sycl.math.copysign %arg0, %arg1 : !sycl_vec_f64_8_
  // CHECK: %{{.*}} = sycl.math.cos %arg0 : !sycl_vec_f64_8_
  %c2 = sycl.math.cos %arg0 : !sycl_vec_f64_8_
  // CHECK: %{{.*}} = sycl.math.exp %arg0 : !sycl_vec_f64_8_
  %e2 = sycl.math.exp %arg0 : !sycl_vec_f64_8_
  // CHECK: %{{.*}} = sycl.math.exp2 %arg0 : !sycl_vec_f64_8_
  %e0 = sycl.math.exp2 %arg0 : !sycl_vec_f64_8_
  // CHECK: %{{.*}} = sycl.math.expm1 %arg0 : !sycl_vec_f64_8_
  %e1 = sycl.math.expm1 %arg0 : !sycl_vec_f64_8_
  // CHECK: %{{.*}} = sycl.math.fabs %arg0 : !sycl_vec_f64_8_
  %f0 = sycl.math.fabs %arg0 : !sycl_vec_f64_8_
  // CHECK: %{{.*}} = sycl.math.floor %arg0 : !sycl_vec_f64_8_
  %f1 = sycl.math.floor %arg0 : !sycl_vec_f64_8_
  // CHECK: %{{.*}} = sycl.math.fma %arg0, %arg1, %arg2 : !sycl_vec_f64_8_
  %f2 = sycl.math.fma %arg0, %arg1, %arg2 : !sycl_vec_f64_8_
  // CHECK: %{{.*}} = sycl.math.log %arg0 : !sycl_vec_f64_8_
  %l2 = sycl.math.log %arg0 : !sycl_vec_f64_8_
  // CHECK: %{{.*}} = sycl.math.log10 %arg0 : !sycl_vec_f64_8_
  %l0 = sycl.math.log10 %arg0 : !sycl_vec_f64_8_
  // CHECK: %{{.*}} = sycl.math.log2 %arg0 : !sycl_vec_f64_8_
  %l1 = sycl.math.log2 %arg0 : !sycl_vec_f64_8_
  // CHECK: %{{.*}} = sycl.math.pow %arg0, %arg1 : !sycl_vec_f64_8_
  %p0 = sycl.math.pow %arg0, %arg1 : !sycl_vec_f64_8_
  // CHECK: %{{.*}} = sycl.math.round %arg0 : !sycl_vec_f64_8_
  %r0 = sycl.math.round %arg0 : !sycl_vec_f64_8_
  // CHECK: %{{.*}} = sycl.math.rsqrt %arg0 : !sycl_vec_f64_8_
  %r1 = sycl.math.rsqrt %arg0 : !sycl_vec_f64_8_
  // CHECK: %{{.*}} = sycl.math.sin %arg0 : !sycl_vec_f64_8_
  %s0 = sycl.math.sin %arg0 : !sycl_vec_f64_8_
  // CHECK: %{{.*}} = sycl.math.sqrt %arg0 : !sycl_vec_f64_8_
  %s1 = sycl.math.sqrt %arg0 : !sycl_vec_f64_8_
  // CHECK: %{{.*}} = sycl.math.trunc %arg0 : !sycl_vec_f64_8_
  %t0 = sycl.math.trunc %arg0 : !sycl_vec_f64_8_

  return
}

!sycl_vec_sycl_half_2_ = !sycl.vec<[!sycl_half, 2], (vector<2xf16>)>

// CHECK-LABEL: test_math_ops_vector_of_half
func.func @test_math_ops_vector_of_half(%arg0 : !sycl_vec_sycl_half_2_, %arg1 : !sycl_vec_sycl_half_2_, %arg2 : !sycl_vec_sycl_half_2_) {
  // CHECK: %{{.*}} = sycl.math.ceil %arg0 : !sycl_vec_sycl_half_2_
  %c0 = sycl.math.ceil %arg0 : !sycl_vec_sycl_half_2_
  // CHECK: %{{.*}} = sycl.math.copysign %arg0, %arg1 : !sycl_vec_sycl_half_2_
  %c1 = sycl.math.copysign %arg0, %arg1 : !sycl_vec_sycl_half_2_
  // CHECK: %{{.*}} = sycl.math.cos %arg0 : !sycl_vec_sycl_half_2_
  %c2 = sycl.math.cos %arg0 : !sycl_vec_sycl_half_2_
  // CHECK: %{{.*}} = sycl.math.exp %arg0 : !sycl_vec_sycl_half_2_
  %e2 = sycl.math.exp %arg0 : !sycl_vec_sycl_half_2_
  // CHECK: %{{.*}} = sycl.math.exp2 %arg0 : !sycl_vec_sycl_half_2_
  %e0 = sycl.math.exp2 %arg0 : !sycl_vec_sycl_half_2_
  // CHECK: %{{.*}} = sycl.math.expm1 %arg0 : !sycl_vec_sycl_half_2_
  %e1 = sycl.math.expm1 %arg0 : !sycl_vec_sycl_half_2_
  // CHECK: %{{.*}} = sycl.math.fabs %arg0 : !sycl_vec_sycl_half_2_
  %f0 = sycl.math.fabs %arg0 : !sycl_vec_sycl_half_2_
  // CHECK: %{{.*}} = sycl.math.floor %arg0 : !sycl_vec_sycl_half_2_
  %f1 = sycl.math.floor %arg0 : !sycl_vec_sycl_half_2_
  // CHECK: %{{.*}} = sycl.math.fma %arg0, %arg1, %arg2 : !sycl_vec_sycl_half_2_
  %f2 = sycl.math.fma %arg0, %arg1, %arg2 : !sycl_vec_sycl_half_2_
  // CHECK: %{{.*}} = sycl.math.log %arg0 : !sycl_vec_sycl_half_2_
  %l2 = sycl.math.log %arg0 : !sycl_vec_sycl_half_2_
  // CHECK: %{{.*}} = sycl.math.log10 %arg0 : !sycl_vec_sycl_half_2_
  %l0 = sycl.math.log10 %arg0 : !sycl_vec_sycl_half_2_
  // CHECK: %{{.*}} = sycl.math.log2 %arg0 : !sycl_vec_sycl_half_2_
  %l1 = sycl.math.log2 %arg0 : !sycl_vec_sycl_half_2_
  // CHECK: %{{.*}} = sycl.math.pow %arg0, %arg1 : !sycl_vec_sycl_half_2_
  %p0 = sycl.math.pow %arg0, %arg1 : !sycl_vec_sycl_half_2_
  // CHECK: %{{.*}} = sycl.math.round %arg0 : !sycl_vec_sycl_half_2_
  %r0 = sycl.math.round %arg0 : !sycl_vec_sycl_half_2_
  // CHECK: %{{.*}} = sycl.math.rsqrt %arg0 : !sycl_vec_sycl_half_2_
  %r1 = sycl.math.rsqrt %arg0 : !sycl_vec_sycl_half_2_
  // CHECK: %{{.*}} = sycl.math.sin %arg0 : !sycl_vec_sycl_half_2_
  %s0 = sycl.math.sin %arg0 : !sycl_vec_sycl_half_2_
  // CHECK: %{{.*}} = sycl.math.sqrt %arg0 : !sycl_vec_sycl_half_2_
  %s1 = sycl.math.sqrt %arg0 : !sycl_vec_sycl_half_2_
  // CHECK: %{{.*}} = sycl.math.trunc %arg0 : !sycl_vec_sycl_half_2_
  %t0 = sycl.math.trunc %arg0 : !sycl_vec_sycl_half_2_

  return
}
