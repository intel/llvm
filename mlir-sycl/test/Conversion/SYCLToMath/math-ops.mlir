// RUN: sycl-mlir-opt -convert-sycl-to-math %s | FileCheck %s

// CHECK-LABEL: test_atan
func.func @test_atan(%arg0: f32) -> (f32) {
  // CHECK: %{{.*}} = math.atan %arg0 : f32
  %0 = sycl.math.atan %arg0 : f32
  return %0 : f32
}
