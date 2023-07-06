// RUN: sycl-mlir-opt %s | sycl-mlir-opt | FileCheck %s
// RUN: sycl-mlir-opt %s --mlir-print-op-generic | sycl-mlir-opt | FileCheck %s

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_nd_range_1_ = !sycl.nd_range<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_accessor_2_i32_r_gb = !sycl.accessor<[2, i32, read, global_buffer], (!sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK-LABEL: test_host_constructor
// CHECK:  %1 = llvm.alloca %0 x !sycl_id_1_ : (i32) -> !llvm.ptr
// CHECK:  sycl.host.constructor(%1) {type = !sycl_id_1_} : (!llvm.ptr) -> ()
func.func @test_host_constructor() -> !llvm.ptr {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !sycl_id_1_ : (i32) -> !llvm.ptr
  sycl.host.constructor(%1) {type = !sycl_id_1_} : (!llvm.ptr) -> ()
  return %1 : !llvm.ptr
}

// CHECK-LABEL: test_host_constructor_args
// CHECK:  %1 = llvm.alloca %0 x !sycl_accessor_2_i32_r_gb : (i32) -> !llvm.ptr
// CHECK:  sycl.host.constructor(%[[#PTR:]], %arg0, %arg1) {type = !sycl_accessor_2_i32_r_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
func.func @test_host_constructor_args(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> !llvm.ptr {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !sycl_accessor_2_i32_r_gb : (i32) -> !llvm.ptr
  sycl.host.constructor(%1, %arg0, %arg1) {type = !sycl_accessor_2_i32_r_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  return %1 : !llvm.ptr
}

gpu.module @kernels {
  gpu.func @k0() kernel {
    gpu.return
  }
}

// CHECK-LABEL:  sycl.host.kernel_name @kernel_ref -> @kernels::@k0
sycl.host.kernel_name @kernel_ref -> @kernels::@k0

// CHECK-LABEL:  func.func @f() -> !llvm.ptr {
// CHECK-NEXT:     %0 = sycl.host.get_kernel @kernels::@k0 : !llvm.ptr
func.func @f() -> !llvm.ptr {
  %0 = sycl.host.get_kernel @kernels::@k0 : !llvm.ptr
  func.return %0 : !llvm.ptr
}

// CHECK-LABEL:  func.func @set_kernel(
// CHECK-SAME:                         %[[VAL_0:.*]]: !llvm.ptr) {
// CHECK-NEXT:     sycl.host.handler.set_kernel %[[VAL_0]] -> @kernels::@k0
func.func @set_kernel(%handler: !llvm.ptr) {
  sycl.host.handler.set_kernel %handler -> @kernels::@k0 : !llvm.ptr
  func.return
}

// CHECK-LABEL:   func.func @set_nd_range(
// CHECK-SAME:                            %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                            %[[VAL_1:.*]]: memref<?x!sycl_nd_range_1_>) {
// CHECK-NEXT:      sycl.host.handler.set_nd_range %[[VAL_0]] -> %[[VAL_1]] : !llvm.ptr, memref<?x!sycl_nd_range_1_>
func.func @set_nd_range(%handler: !llvm.ptr, %nd_range: memref<?x!sycl_nd_range_1_>) {
  sycl.host.handler.set_nd_range %handler -> %nd_range : !llvm.ptr, memref<?x!sycl_nd_range_1_>
  func.return
}

// CHECK-LABEL:   func.func @set_nd_range_range(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                                  %[[VAL_1:.*]]: memref<?x!sycl_range_1_>) {
// CHECK-NEXT:      sycl.host.handler.set_nd_range %[[VAL_0]] -> %[[VAL_1]] : !llvm.ptr, memref<?x!sycl_range_1_>
func.func @set_nd_range_range(%handler: !llvm.ptr, %range: memref<?x!sycl_range_1_>) {
  sycl.host.handler.set_nd_range %handler -> %range : !llvm.ptr, memref<?x!sycl_range_1_>
  func.return
}

// CHECK-LABEL:   func.func @set_nd_range_range_with_offset(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                                              %[[VAL_1:.*]]: memref<?x!sycl_range_1_>,
// CHECK-SAME:                                              %[[VAL_2:.*]]: memref<?x!sycl_id_1_>) {
// CHECK-NEXT:      sycl.host.handler.set_nd_range %[[VAL_0]] -> %[[VAL_1]], %[[VAL_2]] : !llvm.ptr, memref<?x!sycl_range_1_>, memref<?x!sycl_id_1_>
func.func @set_nd_range_range_with_offset(%handler: !llvm.ptr, %range: memref<?x!sycl_range_1_>, %offset: memref<?x!sycl_id_1_>) {
  sycl.host.handler.set_nd_range %handler -> %range, %offset : !llvm.ptr, memref<?x!sycl_range_1_>, memref<?x!sycl_id_1_>
  func.return
}
