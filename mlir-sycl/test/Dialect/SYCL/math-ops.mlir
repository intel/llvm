// RUN: sycl-mlir-opt %s | sycl-mlir-opt | FileCheck %s
// RUN: sycl-mlir-opt %s --mlir-print-op-generic | sycl-mlir-opt | FileCheck %s

!sycl_float2 = !sycl.vec<[f32, 2], (vector<2 x f32>)>

// CHECK-LABEL: test_atan
func.func @test_atan(%arg0: f32, %arg1 : !sycl_float2) -> (f32, !sycl_float2) {
  %0 = sycl.math.atan %arg0 : f32
  %1 = sycl.math.atan %arg1 : !sycl_float2
  return %0, %1 : f32, !sycl_float2
}
