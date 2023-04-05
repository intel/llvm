// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm='use-opaque-pointers=0' -verify-diagnostics %s | FileCheck %s

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
func.func @cast_sycl_range_to_array(%arg0: memref<?x!sycl_range_1_>) -> memref<?x!sycl_array_1_> {
  // CHECK-LABEL: llvm.func @cast_sycl_range_to_array(
  // CHECK-SAME:                                      [[SRC:%.*]]: !llvm.ptr<[[RANGE1:.*]]>) -> !llvm.ptr<[[ARRAY1:.*]]>
  // CHECK-NEXT: [[RES:%.*]] = llvm.bitcast [[SRC]] : !llvm.ptr<[[RANGE1]]> to !llvm.ptr<[[ARRAY1]]>
  // CHECK-NEXT: llvm.return [[RES]] : !llvm.ptr<[[ARRAY1]]>

  %0 = "sycl.cast"(%arg0) : (memref<?x!sycl_range_1_>) -> memref<?x!sycl_array_1_>
  func.return %0 : memref<?x!sycl_array_1_>
}

// -----

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
func.func @cast_sycl_id_to_array(%arg0: memref<?x!sycl_id_1_>) -> memref<?x!sycl_array_1_> {
  // CHECK-LABEL: llvm.func @cast_sycl_id_to_array(
  // CHECK-SAME:                                   [[SRC:%.*]]: !llvm.ptr<[[ID1:.*]]>) -> !llvm.ptr<[[ARRAY1]]>
  // CHECK-NEXT: [[RES:%.*]] = llvm.bitcast [[SRC]] : !llvm.ptr<[[ID1]]> to !llvm.ptr<[[ARRAY1]]>
  // CHECK-NEXT: llvm.return [[RES]] : !llvm.ptr<[[ARRAY1]]>

  %0 = "sycl.cast"(%arg0) : (memref<?x!sycl_id_1_>) -> memref<?x!sycl_array_1_>
  func.return %0: memref<?x!sycl_array_1_>
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>, !llvm.struct<(ptr<i32, 1>)>)>
func.func @cast_sycl_accessor_to_accessor_common(%arg0: memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?x!sycl.accessor_common> {
  // CHECK-LABEL: llvm.func @cast_sycl_accessor_to_accessor_common(
  // CHECK-SAME:                                                   [[SRC:%.*]]: !llvm.ptr<[[ACC1:.*]]>) -> !llvm.ptr<[[COMMON:.*]]>
  // CHECK-NEXT: [[RES:%.*]] = llvm.bitcast [[SRC]] : !llvm.ptr<[[ACC1]]> to !llvm.ptr<[[COMMON]]>
  // CHECK-NEXT: llvm.return [[RES]] : !llvm.ptr<[[COMMON]]>

  %0 = "sycl.cast"(%arg0) : (memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?x!sycl.accessor_common>
  func.return %0: memref<?x!sycl.accessor_common>
}

!sycl_LocalAccessorBaseDevice_1_ = !sycl.LocalAccessorBaseDevice<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_local_accessor_base_1_i32_rw = !sycl.local_accessor_base<[1, i32, read_write], (!sycl_LocalAccessorBaseDevice_1_, memref<?xi32, 3>)>
func.func @cast_sycl_accessor_to_local_accessor_base(%arg0: memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?x!sycl_local_accessor_base_1_i32_rw> {
  // CHECK-LABEL: llvm.func @cast_sycl_accessor_to_local_accessor_base(
  // CHECK-SAME:                                                       [[SRC:%.*]]: !llvm.ptr<[[ACC1]]>) -> !llvm.ptr<[[LOCALBASE:.*]]>
  // CHECK-NEXT: [[RES:%.*]] = llvm.bitcast [[SRC]] : !llvm.ptr<[[ACC1]]> to !llvm.ptr<[[LOCALBASE]]>
  // CHECK-NEXT: llvm.return [[RES]] : !llvm.ptr<[[LOCALBASE]]>

  %0 = "sycl.cast"(%arg0) : (memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?x!sycl_local_accessor_base_1_i32_rw>
  func.return %0: memref<?x!sycl_local_accessor_base_1_i32_rw>
}

func.func @cast_sycl_accessor_to_owner_less_base(%arg0: memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?x!sycl.owner_less_base> {
  // CHECK-LABEL: llvm.func @cast_sycl_accessor_to_owner_less_base(
  // CHECK-SAME:                                                   [[SRC:%.*]]: !llvm.ptr<[[ACC1]]>) -> !llvm.ptr<[[OWNERLESSBASE:.*]]>
  // CHECK-NEXT: [[RES:%.*]] = llvm.bitcast [[SRC]] : !llvm.ptr<[[ACC1]]> to !llvm.ptr<[[OWNERLESSBASE]]>
  // CHECK-NEXT: llvm.return [[RES]] : !llvm.ptr<[[OWNERLESSBASE]]>
  
  %0 = "sycl.cast"(%arg0) : (memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?x!sycl.owner_less_base>
  func.return %0: memref<?x!sycl.owner_less_base>
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_LocalAccessorBaseDevice_1_ = !sycl.LocalAccessorBaseDevice<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_local_accessor_base_1_i32_rw = !sycl.local_accessor_base<[1, i32, read_write], (!sycl_LocalAccessorBaseDevice_1_, memref<?xi32, 3>)>
func.func @cast_sycl_local_accessor_base_to_accessor_common(%arg0: memref<?x!sycl_local_accessor_base_1_i32_rw>) -> memref<?x!sycl.accessor_common> {
  // CHECK-LABEL: llvm.func @cast_sycl_local_accessor_base_to_accessor_common(
  // CHECK-SAME:                                                              [[SRC:%.*]]: !llvm.ptr<[[LAB1:.*]]>) -> !llvm.ptr<[[COMMON]]>
  // CHECK-NEXT: [[RES:%.*]] = llvm.bitcast [[SRC]] : !llvm.ptr<[[LAB1]]> to !llvm.ptr<[[COMMON]]>
  // CHECK-NEXT: llvm.return [[RES]] : !llvm.ptr<[[COMMON]]
  %0 = "sycl.cast"(%arg0) : (memref<?x!sycl_local_accessor_base_1_i32_rw>) -> memref<?x!sycl.accessor_common>
  func.return %0: memref<?x!sycl.accessor_common>
}

!sycl_local_accessor_1_i32_rw = !sycl.local_accessor<[1, i32], (!sycl_local_accessor_base_1_i32_rw)>
func.func @cast_sycl_local_accessor_to_local_accessor_base(%arg0: memref<?x!sycl_local_accessor_1_i32_rw>) -> memref<?x!sycl_local_accessor_base_1_i32_rw> {
  // CHECK-LABEL: llvm.func @cast_sycl_local_accessor_to_local_accessor_base(
  // CHECK-SAME:                                                             [[SRC:%.*]]: !llvm.ptr<[[LA1:.*]]>) -> !llvm.ptr<[[LAB1]]>
  // CHECK-NEXT: [[RES:%.*]] = llvm.bitcast [[SRC]] : !llvm.ptr<[[LA1]]> to !llvm.ptr<[[LAB1]]>
  // CHECK-NEXT: llvm.return [[RES]] : !llvm.ptr<[[LAB1]]

  %0 = "sycl.cast"(%arg0) : (memref<?x!sycl_local_accessor_1_i32_rw>) -> memref<?x!sycl_local_accessor_base_1_i32_rw>
  func.return %0: memref<?x!sycl_local_accessor_base_1_i32_rw>  
}


