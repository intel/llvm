// RUN: sycl-mlir-opt %s | sycl-mlir-opt | FileCheck %s
// RUN: sycl-mlir-opt %s --mlir-print-op-generic | sycl-mlir-opt | FileCheck %s

// CHECK-LABEL: test_sqrt
func.func @test_sqrt(%arg0: f32) -> (f32) {
  %0 = sycl.math.sqrt %arg0 : f32
  return %0 : f32
}
