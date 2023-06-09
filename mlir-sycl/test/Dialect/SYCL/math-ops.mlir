// RUN: sycl-mlir-opt %s | sycl-mlir-opt | FileCheck %s
// RUN: sycl-mlir-opt %s --mlir-print-op-generic | sycl-mlir-opt | FileCheck %s

// CHECK-LABEL: test_atan
func.func @test_atan(%arg0: f32) -> (f32) {
  %0 = sycl.math.atan %arg0 : f32
  return %0 : f32
}
