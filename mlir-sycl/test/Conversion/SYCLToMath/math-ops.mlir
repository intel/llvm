// RUN: sycl-mlir-opt -convert-sycl-to-math %s | FileCheck %s

// CHECK-LABEL: test_sqrt
func.func @test_sqrt(%arg0: f32) -> (f32) {
  // CHECK: %{{.*}} = math.sqrt %arg0 : f32
  %0 = sycl.math.sqrt %arg0 : f32
  return %0 : f32
}
