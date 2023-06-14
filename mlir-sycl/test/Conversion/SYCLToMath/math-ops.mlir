// RUN: sycl-mlir-opt -convert-sycl-to-math %s | FileCheck %s

// CHECK-LABEL: test_math_ops_scalar
func.func @test_math_ops_scalar(%arg0 : f32, %arg1 : f32, %arg2 : f32) {
  // CHECK: %{{.*}} = math.ceil %arg0 : f32
  %c0 = sycl.math.ceil %arg0 : f32
  // CHECK: %{{.*}} = math.copysign %arg0, %arg1 : f32
  %c1 = sycl.math.copysign %arg0, %arg1 : f32
  // CHECK: %{{.*}} = math.cos %arg0 : f32
  %c2 = sycl.math.cos %arg0 : f32
  // CHECK: %{{.*}} = math.exp2 %arg0 : f32
  %e0 = sycl.math.exp2 %arg0 : f32
  // CHECK: %{{.*}} = math.expm1 %arg0 : f32
  %e1 = sycl.math.expm1 %arg0 : f32
  // CHECK: %{{.*}} = math.exp %arg0 : f32
  %e2 = sycl.math.exp %arg0 : f32
  // CHECK: %{{.*}} = math.absf %arg0 : f32
  %f0 = sycl.math.fabs %arg0 : f32
  // CHECK: %{{.*}} = math.floor %arg0 : f32
  %f1 = sycl.math.floor %arg0 : f32
  // CHECK: %{{.*}} = math.fma %arg0, %arg1, %arg2 : f32
  %f2 = sycl.math.fma %arg0, %arg1, %arg2 : f32
  // CHECK: %{{.*}} = math.log10 %arg0 : f32
  %l0 = sycl.math.log10 %arg0 : f32
  // CHECK: %{{.*}} = math.log2 %arg0 : f32
  %l1 = sycl.math.log2 %arg0 : f32
  // CHECK: %{{.*}} = math.log %arg0 : f32
  %l2 = sycl.math.log %arg0 : f32
  // CHECK: %{{.*}} = math.powf %arg0, %arg1 : f32
  %p0 = sycl.math.pow %arg0, %arg1 : f32
  // CHECK: %{{.*}} = math.round %arg0 : f32
  %r0 = sycl.math.round %arg0 : f32
  // CHECK: %{{.*}} = math.rsqrt %arg0 : f32
  %r1 = sycl.math.rsqrt %arg0 : f32
  // CHECK: %{{.*}} = math.sin %arg0 : f32
  %s0 = sycl.math.sin %arg0 : f32
  // CHECK: %{{.*}} = math.sqrt %arg0 : f32
  %s1 = sycl.math.sqrt %arg0 : f32
  // CHECK: %{{.*}} = math.trunc %arg0 : f32
  %t0 = sycl.math.trunc %arg0 : f32

  return
}
