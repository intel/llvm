// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm="use-bare-ptr-call-conv" %s | FileCheck %s
// XFAIL: *

// Failing because signature is not recognized

//===-------------------------------------------------------------------------------------------------===//
// sycl.accessor.subscript with id offset and atomic return type
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_ato_gb = !sycl.accessor<[1, i32, atomic, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 3>)>)>
!sycl_atomic_i32_1_ = !sycl.atomic<[i32, 1], (memref<?xi32, 1>)>

func.func @test(%acc: memref<?x!sycl_accessor_1_i32_ato_gb>, %idx: memref<?x!sycl_id_1_>) -> !sycl_atomic_i32_1_ {
  %0 = sycl.accessor.subscript %acc[%idx] { ArgumentTypes = [memref<?x!sycl_accessor_1_i32_ato_gb>, memref<?x!sycl_id_1_>], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"id" }  : (memref<?x!sycl_accessor_1_i32_ato_gb>, memref<?x!sycl_id_1_>) -> !sycl_atomic_i32_1_
  return %0 : !sycl_atomic_i32_1_
}
