// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm="use-bare-ptr-call-conv" -verify-diagnostics %s | FileCheck %s
// XFAIL: *

// Failing because it is not implemented

//===-------------------------------------------------------------------------------------------------===//
// sycl.range.get with memref result type
//===-------------------------------------------------------------------------------------------------===//

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

func.func @test(%range: !sycl_range_3_, %idx: i32) -> memref<?xi64> {
  %0 = "sycl.range.get"(%range, %idx) { ArgumentTypes = [!sycl_range_3_, i32], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"range" }  : (!sycl_range_3_, i32) -> memref<?xi64>
  return %0 : memref<?xi64>
}
