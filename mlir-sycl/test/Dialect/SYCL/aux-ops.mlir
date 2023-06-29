// RUN: sycl-mlir-opt %s | sycl-mlir-opt | FileCheck %s
// RUN: sycl-mlir-opt %s --mlir-print-op-generic | sycl-mlir-opt | FileCheck %s
// RUN: sycl-mlir-opt %s -canonicalize | FileCheck %s --check-prefix=FOLD

!sycl_half = !sycl.half<(f16)>

// CHECK-LABEL: test_half
func.func @test_half(%arg0 : f16, %arg1 : !sycl_half) -> (!sycl_half, f16) {
  // CHECK: %{{.*}} = sycl.mlir.wrap %arg0 : f16 to !sycl_half
  %0 = sycl.mlir.wrap %arg0 : f16 to !sycl_half
  // CHECK: %{{.*}} = sycl.mlir.unwrap %arg1 : !sycl_half to f16
  %1 = sycl.mlir.unwrap %arg1 : !sycl_half to f16
  
  return %0, %1 : !sycl_half, f16
}

// FOLD-LABEL: test_folder
func.func @test_folder(%arg0 : f16, %arg1 : !sycl_half) -> (!sycl_half, f16) {
  %0 = sycl.mlir.wrap %arg0 : f16 to !sycl_half
  %1 = sycl.mlir.unwrap %0 : !sycl_half to f16
  // FOLD: %[[wrap:.*]] = sycl.mlir.wrap %arg0 : f16 to !sycl_half
  %2 = sycl.mlir.wrap %1 : f16 to !sycl_half

  %3 = sycl.mlir.unwrap %arg1 : !sycl_half to f16
  %4 = sycl.mlir.wrap %3 : f16 to !sycl_half
  // FOLD: %[[unwrap:.*]] = sycl.mlir.unwrap %arg1 : !sycl_half to f16
  %5 = sycl.mlir.unwrap %4 : !sycl_half to f16
  
  // FOLD: return %[[wrap]], %[[unwrap]] : !sycl_half, f16
  return %2, %5 : !sycl_half, f16
}
