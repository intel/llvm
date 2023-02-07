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

// -----

// Failing because this signature is not recognized as legal.

//===-------------------------------------------------------------------------------------------------===//
// sycl.accessor.subscript with atomic access mode and sycl.id offset
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_atomic_i32_1_ = !sycl.atomic<[i32, 1], (memref<?xi32, 1>)>

func.func @test(%acc: !sycl_accessor_1_i32_rw_gb, %off: i64) -> !sycl_atomic_i32_1_ {
  %0 = sycl.accessor.subscript %acc[%off] { ArgumentTypes = [!sycl_accessor_1_i32_rw_gb, i64], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"accessor" }  : (!sycl_accessor_1_i32_rw_gb, i64) -> !sycl_atomic_i32_1_
  return %0 : !sycl_atomic_i32_1_
}
