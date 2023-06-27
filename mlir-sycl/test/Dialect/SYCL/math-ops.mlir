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
