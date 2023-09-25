// RUN: sycl-mlir-opt %s | sycl-mlir-opt | FileCheck %s
// RUN: sycl-mlir-opt %s --mlir-print-op-generic | sycl-mlir-opt | FileCheck %s

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_nd_range_1_ = !sycl.nd_range<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_accessor_2_i32_r_dev = !sycl.accessor<[2, i32, read, device], (!sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>, !llvm.struct<(ptr<i32, 1>)>)>

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
// CHECK:  %1 = llvm.alloca %0 x !sycl_accessor_2_i32_r_dev : (i32) -> !llvm.ptr
// CHECK:  sycl.host.constructor(%[[#PTR:]], %arg0, %arg1) {type = !sycl_accessor_2_i32_r_dev} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
func.func @test_host_constructor_args(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> !llvm.ptr {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !sycl_accessor_2_i32_r_dev : (i32) -> !llvm.ptr
  sycl.host.constructor(%1, %arg0, %arg1) {type = !sycl_accessor_2_i32_r_dev} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  return %1 : !llvm.ptr
}

gpu.module @kernels {
  gpu.func @k0() kernel {
    gpu.return
  }
}

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
// CHECK-SAME:                            %[[VAL_1:.*]]: !llvm.ptr) {
// CHECK-NEXT:      sycl.host.handler.set_nd_range %[[VAL_0]] -> nd_range %[[VAL_1]] : !llvm.ptr, !llvm.ptr
func.func @set_nd_range(%handler: !llvm.ptr, %nd_range: !llvm.ptr) {
  sycl.host.handler.set_nd_range %handler -> nd_range %nd_range : !llvm.ptr, !llvm.ptr
  func.return
}

// CHECK-LABEL:   func.func @set_nd_range_range(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                                  %[[VAL_1:.*]]: !llvm.ptr) {
// CHECK-NEXT:      sycl.host.handler.set_nd_range %[[VAL_0]] -> range %[[VAL_1]] : !llvm.ptr, !llvm.ptr
func.func @set_nd_range_range(%handler: !llvm.ptr, %range: !llvm.ptr) {
  sycl.host.handler.set_nd_range %handler -> range %range : !llvm.ptr, !llvm.ptr
  func.return
}

// CHECK-LABEL:   func.func @set_nd_range_range_with_offset(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: !llvm.ptr, %[[VAL_2:.*]]: !llvm.ptr) {
// CHECK-NEXT:      sycl.host.handler.set_nd_range %[[VAL_0]] -> range %[[VAL_1]], offset %[[VAL_2]] : !llvm.ptr, !llvm.ptr, !llvm.ptr
func.func @set_nd_range_range_with_offset(%handler: !llvm.ptr, %range: !llvm.ptr, %offset: !llvm.ptr) {
  sycl.host.handler.set_nd_range %handler -> range %range, offset %offset : !llvm.ptr, !llvm.ptr, !llvm.ptr
  func.return
}

// CHECK-LABEL:   func.func @set_captured(
// CHECK-SAME:                            %[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: i16, %[[VAL_2:.*]]: !llvm.ptr, %[[VAL_3:.*]]: !llvm.ptr) {
// CHECK:           sycl.host.set_captured %[[VAL_0]][0] = %[[VAL_1]] : !llvm.ptr, i16
// CHECK:           sycl.host.set_captured %[[VAL_0]][1] = %[[VAL_2]] : !llvm.ptr, !llvm.ptr
// CHECK:           sycl.host.set_captured %[[VAL_0]][2] = %[[VAL_3]] : !llvm.ptr, !llvm.ptr (!sycl_accessor_2_i32_r_dev)
func.func @set_captured(%lambda: !llvm.ptr, %scalar_arg: i16, %struct_arg: !llvm.ptr, %accessor_arg: !llvm.ptr) {
  sycl.host.set_captured %lambda[0] = %scalar_arg : !llvm.ptr, i16
  sycl.host.set_captured %lambda[1] = %struct_arg : !llvm.ptr, !llvm.ptr
  sycl.host.set_captured %lambda[2] = %accessor_arg : !llvm.ptr,  !llvm.ptr (!sycl_accessor_2_i32_r_dev)
  func.return
}

// CHECK-LABEL:   func.func @schedule_kernel_single_task(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: !llvm.ptr) {
// CHECK:           sycl.host.schedule_kernel %[[VAL_0]] -> @kernels::@k0 : (!llvm.ptr) -> ()
// CHECK:           sycl.host.schedule_kernel %[[VAL_0]] -> @kernels::@k0(%[[VAL_1]]) : (!llvm.ptr, i32) -> ()
// CHECK:           sycl.host.schedule_kernel %[[VAL_0]] -> @kernels::@k0(%[[VAL_1]], %[[VAL_2]]) : (!llvm.ptr, i32, i32) -> ()
// CHECK:           sycl.host.schedule_kernel %[[VAL_0]] -> @kernels::@k0(%[[VAL_1]], %[[VAL_2]], %[[VAL_3]]: !sycl_accessor_2_i32_r_dev) : (!llvm.ptr, i32, i32, !llvm.ptr) -> ()
// CHECK:           return
// CHECK:         }
func.func @schedule_kernel_single_task(%handler: !llvm.ptr, %arg0: i32, %arg1: i32, %arg2: !llvm.ptr) {
  sycl.host.schedule_kernel %handler -> @kernels::@k0 : (!llvm.ptr) -> ()
  sycl.host.schedule_kernel %handler -> @kernels::@k0(%arg0) : (!llvm.ptr, i32) -> ()
  sycl.host.schedule_kernel %handler -> @kernels::@k0(%arg0, %arg1) : (!llvm.ptr, i32, i32) -> ()
  sycl.host.schedule_kernel %handler -> @kernels::@k0(%arg0, %arg1, %arg2: !sycl_accessor_2_i32_r_dev) : (!llvm.ptr, i32, i32, !llvm.ptr) -> ()
  func.return
}

// CHECK-LABEL:   func.func @schedule_kernel_nd_range(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: !llvm.ptr, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: !llvm.ptr) {
// CHECK:           sycl.host.schedule_kernel %[[VAL_0]] -> @kernels::@k0[nd_range %[[VAL_1]]] : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:           sycl.host.schedule_kernel %[[VAL_0]] -> @kernels::@k0[nd_range %[[VAL_1]]](%[[VAL_2]]) : (!llvm.ptr, !llvm.ptr, i32) -> ()
// CHECK:           sycl.host.schedule_kernel %[[VAL_0]] -> @kernels::@k0[nd_range %[[VAL_1]]](%[[VAL_2]], %[[VAL_3]]) : (!llvm.ptr, !llvm.ptr, i32, i32) -> ()
// CHECK:           sycl.host.schedule_kernel %[[VAL_0]] -> @kernels::@k0[nd_range %[[VAL_1]]](%[[VAL_2]], %[[VAL_3]], %[[VAL_4]]: !sycl_accessor_2_i32_r_dev) : (!llvm.ptr, !llvm.ptr, i32, i32, !llvm.ptr) -> ()
// CHECK:           return
// CHECK:         }
func.func @schedule_kernel_nd_range(%handler: !llvm.ptr, %nd_range: !llvm.ptr, %arg0: i32, %arg1: i32, %arg2: !llvm.ptr) {
  sycl.host.schedule_kernel %handler -> @kernels::@k0[nd_range %nd_range] : (!llvm.ptr, !llvm.ptr) -> ()
  sycl.host.schedule_kernel %handler -> @kernels::@k0[nd_range %nd_range](%arg0) : (!llvm.ptr, !llvm.ptr, i32) -> ()
  sycl.host.schedule_kernel %handler -> @kernels::@k0[nd_range %nd_range](%arg0, %arg1) : (!llvm.ptr, !llvm.ptr, i32, i32) -> ()
  sycl.host.schedule_kernel %handler -> @kernels::@k0[nd_range %nd_range](%arg0, %arg1, %arg2: !sycl_accessor_2_i32_r_dev) : (!llvm.ptr, !llvm.ptr, i32, i32, !llvm.ptr) -> ()
  func.return
}

// CHECK-LABEL:   func.func @schedule_kernel_range(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: !llvm.ptr, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: !llvm.ptr) {
// CHECK:           sycl.host.schedule_kernel %[[VAL_0]] -> @kernels::@k0[range %[[VAL_1]]] : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK:           sycl.host.schedule_kernel %[[VAL_0]] -> @kernels::@k0[range %[[VAL_1]]](%[[VAL_2]]) : (!llvm.ptr, !llvm.ptr, i32) -> ()
// CHECK:           sycl.host.schedule_kernel %[[VAL_0]] -> @kernels::@k0[range %[[VAL_1]]](%[[VAL_2]], %[[VAL_3]]) : (!llvm.ptr, !llvm.ptr, i32, i32) -> ()
// CHECK:           sycl.host.schedule_kernel %[[VAL_0]] -> @kernels::@k0[range %[[VAL_1]]](%[[VAL_2]], %[[VAL_3]], %[[VAL_4]]: !sycl_accessor_2_i32_r_dev) : (!llvm.ptr, !llvm.ptr, i32, i32, !llvm.ptr) -> ()
// CHECK:           return
// CHECK:         }
func.func @schedule_kernel_range(%handler: !llvm.ptr, %range: !llvm.ptr, %arg0: i32, %arg1: i32, %arg2: !llvm.ptr) {
  sycl.host.schedule_kernel %handler -> @kernels::@k0[range %range] : (!llvm.ptr, !llvm.ptr) -> ()
  sycl.host.schedule_kernel %handler -> @kernels::@k0[range %range](%arg0) : (!llvm.ptr, !llvm.ptr, i32) -> ()
  sycl.host.schedule_kernel %handler -> @kernels::@k0[range %range](%arg0, %arg1) : (!llvm.ptr, !llvm.ptr, i32, i32) -> ()
  sycl.host.schedule_kernel %handler -> @kernels::@k0[range %range](%arg0, %arg1, %arg2: !sycl_accessor_2_i32_r_dev) : (!llvm.ptr, !llvm.ptr, i32, i32, !llvm.ptr) -> ()
  func.return
}

// CHECK-LABEL:   func.func @schedule_kernel_range_with_offset(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: !llvm.ptr, %[[VAL_2:.*]]: !llvm.ptr, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: !llvm.ptr) {
// CHECK:           sycl.host.schedule_kernel %[[VAL_0]] -> @kernels::@k0[range %[[VAL_1]], offset %[[VAL_2]]] : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:           sycl.host.schedule_kernel %[[VAL_0]] -> @kernels::@k0[range %[[VAL_1]], offset %[[VAL_2]]](%[[VAL_3]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
// CHECK:           sycl.host.schedule_kernel %[[VAL_0]] -> @kernels::@k0[range %[[VAL_1]], offset %[[VAL_2]]](%[[VAL_3]], %[[VAL_4]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> ()
// CHECK:           sycl.host.schedule_kernel %[[VAL_0]] -> @kernels::@k0[range %[[VAL_1]], offset %[[VAL_2]]](%[[VAL_3]], %[[VAL_4]], %[[VAL_5]]: !sycl_accessor_2_i32_r_dev) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, !llvm.ptr) -> ()
// CHECK:           return
// CHECK:         }
func.func @schedule_kernel_range_with_offset(%handler: !llvm.ptr, %range: !llvm.ptr, %offset: !llvm.ptr, %arg0: i32, %arg1: i32, %arg2: !llvm.ptr) {
  sycl.host.schedule_kernel %handler -> @kernels::@k0[range %range, offset %offset] : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.schedule_kernel %handler -> @kernels::@k0[range %range, offset %offset](%arg0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
  sycl.host.schedule_kernel %handler -> @kernels::@k0[range %range, offset %offset](%arg0, %arg1) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> ()
  sycl.host.schedule_kernel %handler -> @kernels::@k0[range %range, offset %offset](%arg0, %arg1, %arg2: !sycl_accessor_2_i32_r_dev) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, !llvm.ptr) -> ()
  func.return
}

llvm.func internal @cgf(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  llvm.return
}

// CHECK-LABEL:   func.func @submit_kernel(
// CHECK-SAME:                             %[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: !llvm.ptr, %[[VAL_2:.*]]: !llvm.ptr, %[[VAL_3:.*]]: !llvm.ptr) {
// CHECK:           sycl.host.submit %[[VAL_1]]({{\[}}%[[VAL_2]], %[[VAL_3]]]@cgf) -> %[[VAL_0]] : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr
// CHECK:           return
// CHECK:         }
func.func @submit_kernel(%event: !llvm.ptr, %queue: !llvm.ptr,
                         %arg0: !llvm.ptr, %arg1 : !llvm.ptr) {
  sycl.host.submit %queue([%arg0, %arg1]@cgf) -> %event
    : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr
  func.return
}

// CHECK-LABEL:   func.func @submit_kernel_no_args(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !llvm.ptr) {
// CHECK:           sycl.host.submit %[[VAL_1]](@cgf) -> %[[VAL_0]] : !llvm.ptr, !llvm.ptr
// CHECK:           return
// CHECK:         }
func.func @submit_kernel_no_args(%event: !llvm.ptr, %queue: !llvm.ptr) {
  sycl.host.submit %queue(@cgf) -> %event : !llvm.ptr, !llvm.ptr
  func.return
}
