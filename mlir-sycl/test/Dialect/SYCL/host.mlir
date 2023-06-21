// RUN: sycl-mlir-opt %s | sycl-mlir-opt | FileCheck %s
// RUN: sycl-mlir-opt %s --mlir-print-op-generic | sycl-mlir-opt | FileCheck %s

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_accessor_2_i32_r_gb = !sycl.accessor<[2, i32, read, global_buffer], (!sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK-LABEL: test_host_constructor
// CHECK-NEXT:  sycl.host.constructor() {type = !sycl_id_1_} : () -> !llvm.ptr
func.func @test_host_constructor() -> !llvm.ptr {
  %0 = sycl.host.constructor() {type = !sycl_id_1_} : () -> !llvm.ptr
  return %0 : !llvm.ptr
}

// CHECK-LABEL: test_host_constructor_args
// CHECK-NEXT:  sycl.host.constructor(%arg0, %arg1) {type = !sycl_accessor_2_i32_r_gb} : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
func.func @test_host_constructor_args(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> !llvm.ptr {
  %0 = sycl.host.constructor(%arg0, %arg1) {type = !sycl_accessor_2_i32_r_gb} : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  return %0 : !llvm.ptr
}
