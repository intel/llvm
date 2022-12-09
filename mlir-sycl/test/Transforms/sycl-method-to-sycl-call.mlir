// RUN: sycl-mlir-opt -sycl-method-to-sycl-call -split-input-file -verify-diagnostics %s | FileCheck %s

!sycl_array_2_ = !sycl.array<[2], (memref<2xi64, 4>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl_array_2_)>
!sycl_range_2_ = !sycl.range<[2], (!sycl_array_2_)>
!sycl_accessor_2_i32_rw_gb = !sycl.accessor<[2, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>)>

// CHECK-LABEL:   func.func @accessor_subscript_operator(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !sycl_accessor_2_i32_rw_gb,
// CHECK-SAME:                                           %[[VAL_1:.*]]: memref<?x!sycl_id_2_>) -> memref<?xi32, 4> {
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_3:.*]] = memref.alloca() : memref<1x!sycl_accessor_2_i32_rw_gb>
// CHECK-NEXT:      %[[VAL_4:.*]] = memref.alloca() : memref<1xindex>
// CHECK-NEXT:      memref.store %[[VAL_0]], %[[VAL_3]]{{\[}}%[[VAL_2]]] : memref<1x!sycl_accessor_2_i32_rw_gb>
// CHECK-NEXT:      memref.store %[[VAL_2]], %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<1xindex>
// CHECK-NEXT:      %[[VAL_5:.*]] = memref.reshape %[[VAL_3]](%[[VAL_4]]) : (memref<1x!sycl_accessor_2_i32_rw_gb>, memref<1xindex>) -> memref<0x!sycl_accessor_2_i32_rw_gb>
// CHECK-NEXT:      %[[VAL_6:.*]] = "polygeist.memref2pointer"(%[[VAL_5]]) : (memref<0x!sycl_accessor_2_i32_rw_gb>) -> !llvm.ptr<!sycl_accessor_2_i32_rw_gb>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.addrspacecast %[[VAL_6]] : !llvm.ptr<!sycl_accessor_2_i32_rw_gb> to !llvm.ptr<!sycl_accessor_2_i32_rw_gb, 4>
// CHECK-NEXT:      %[[VAL_8:.*]] = "polygeist.pointer2memref"(%[[VAL_7]]) : (!llvm.ptr<!sycl_accessor_2_i32_rw_gb, 4>) -> memref<?x!sycl_accessor_2_i32_rw_gb, 4>
// CHECK-NEXT:      %[[VAL_9:.*]] = sycl.call(%[[VAL_8]], %[[VAL_1]]) {FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi2EvEERiNS0_2idILi2EEE, TypeName = @accessor} : (memref<?x!sycl_accessor_2_i32_rw_gb, 4>, memref<?x!sycl_id_2_>) -> memref<?xi32, 4>
// CHECK-NEXT:      return %[[VAL_9]] : memref<?xi32, 4>
// CHECK-NEXT:    }

func.func @accessor_subscript_operator(%arg0: !sycl_accessor_2_i32_rw_gb, %arg1: memref<?x!sycl_id_2_>) -> memref<?xi32, 4> {
  %0 = sycl.accessor.subscript %arg0[%arg1] {BaseType = memref<?x!sycl_accessor_2_i32_rw_gb, 4>, FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V18accessorIiLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi2EvEERiNS0_2idILi2EEE, TypeName = @accessor} : (!sycl_accessor_2_i32_rw_gb, memref<?x!sycl_id_2_>) -> memref<?xi32, 4>
  return %0 : memref<?xi32, 4>
}

// CHECK-LABEL:   func.func @range_get(
// CHECK-SAME:                         %[[VAL_0:.*]]: !sycl_range_2_,
// CHECK-SAME:                         %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_3:.*]] = memref.alloca() : memref<1x!sycl_range_2_>
// CHECK-NEXT:      %[[VAL_4:.*]] = memref.alloca() : memref<1xindex>
// CHECK-NEXT:      memref.store %[[VAL_0]], %[[VAL_3]]{{\[}}%[[VAL_2]]] : memref<1x!sycl_range_2_>
// CHECK-NEXT:      memref.store %[[VAL_2]], %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<1xindex>
// CHECK-NEXT:      %[[VAL_5:.*]] = memref.reshape %[[VAL_3]](%[[VAL_4]]) : (memref<1x!sycl_range_2_>, memref<1xindex>) -> memref<0x!sycl_range_2_>
// CHECK-NEXT:      %[[VAL_6:.*]] = "polygeist.memref2pointer"(%[[VAL_5]]) : (memref<0x!sycl_range_2_>) -> !llvm.ptr<!sycl_range_2_>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.addrspacecast %[[VAL_6]] : !llvm.ptr<!sycl_range_2_> to !llvm.ptr<!sycl_range_2_, 4>
// CHECK-NEXT:      %[[VAL_8:.*]] = "polygeist.pointer2memref"(%[[VAL_7]]) : (!llvm.ptr<!sycl_range_2_, 4>) -> memref<?x!sycl_range_2_, 4>
// CHECK-NEXT:      %[[VAL_9:.*]] = sycl.cast(%[[VAL_8]]) : (memref<?x!sycl_range_2_, 4>) -> memref<?x!sycl_array_2_, 4>
// CHECK-NEXT:      %[[VAL_10:.*]] = sycl.call(%[[VAL_9]], %[[VAL_1]]) {FunctionName = @get, MangledFunctionName = @_ZNK4sycl3_V16detail5arrayILi2EE3getEi, TypeName = @array} : (memref<?x!sycl_array_2_, 4>, i32) -> i64
// CHECK-NEXT:      return %[[VAL_10]] : i64
// CHECK-NEXT:    }

func.func @range_get(%arg0: !sycl_range_2_, %arg1: i32) -> i64 {
  %0 = "sycl.range.get"(%arg0, %arg1) {BaseType = memref<?x!sycl_array_2_, 4>, FunctionName = @get, MangledFunctionName = @_ZNK4sycl3_V16detail5arrayILi2EE3getEi, TypeName = @array} : (!sycl_range_2_, i32) -> i64
  return %0 : i64
}

// CHECK-LABEL:   func.func @range_size(
// CHECK-SAME:                          %[[VAL_0:.*]]: !sycl_range_2_) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_2:.*]] = memref.alloca() : memref<1x!sycl_range_2_>
// CHECK-NEXT:      %[[VAL_3:.*]] = memref.alloca() : memref<1xindex>
// CHECK-NEXT:      memref.store %[[VAL_0]], %[[VAL_2]]{{\[}}%[[VAL_1]]] : memref<1x!sycl_range_2_>
// CHECK-NEXT:      memref.store %[[VAL_1]], %[[VAL_3]]{{\[}}%[[VAL_1]]] : memref<1xindex>
// CHECK-NEXT:      %[[VAL_4:.*]] = memref.reshape %[[VAL_2]](%[[VAL_3]]) : (memref<1x!sycl_range_2_>, memref<1xindex>) -> memref<0x!sycl_range_2_>
// CHECK-NEXT:      %[[VAL_5:.*]] = "polygeist.memref2pointer"(%[[VAL_4]]) : (memref<0x!sycl_range_2_>) -> !llvm.ptr<!sycl_range_2_>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.addrspacecast %[[VAL_5]] : !llvm.ptr<!sycl_range_2_> to !llvm.ptr<!sycl_range_2_, 4>
// CHECK-NEXT:      %[[VAL_7:.*]] = "polygeist.pointer2memref"(%[[VAL_6]]) : (!llvm.ptr<!sycl_range_2_, 4>) -> memref<?x!sycl_range_2_, 4>
// CHECK-NEXT:      %[[VAL_8:.*]] = sycl.call(%[[VAL_7]]) {FunctionName = @size, MangledFunctionName = @_ZNK4sycl3_V15rangeILi2EE4sizeEv, TypeName = @range} : (memref<?x!sycl_range_2_, 4>) -> i64
// CHECK-NEXT:      return %[[VAL_8]] : i64
// CHECK-NEXT:    }

func.func @range_size(%arg0: !sycl_range_2_) -> i64 {
  %0 = "sycl.range.size"(%arg0) {BaseType = memref<?x!sycl_range_2_, 4>, FunctionName = @size, MangledFunctionName = @_ZNK4sycl3_V15rangeILi2EE4sizeEv, TypeName = @range} : (!sycl_range_2_) -> i64
  return %0 : i64
}

// CHECK-LABEL:   func.func @sycl_item_get_id(
// CHECK-SAME:                                %[[VAL_0:.*]]: !sycl_item_1_) -> !sycl_id_1_ {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_2:.*]] = memref.alloca() : memref<1x!sycl_item_1_>
// CHECK-NEXT:      %[[VAL_3:.*]] = memref.alloca() : memref<1xindex>
// CHECK-NEXT:      memref.store %[[VAL_0]], %[[VAL_2]]{{\[}}%[[VAL_1]]] : memref<1x!sycl_item_1_>
// CHECK-NEXT:      memref.store %[[VAL_1]], %[[VAL_3]]{{\[}}%[[VAL_1]]] : memref<1xindex>
// CHECK-NEXT:      %[[VAL_4:.*]] = memref.reshape %[[VAL_2]](%[[VAL_3]]) : (memref<1x!sycl_item_1_>, memref<1xindex>) -> memref<0x!sycl_item_1_>
// CHECK-NEXT:      %[[VAL_5:.*]] = "polygeist.memref2pointer"(%[[VAL_4]]) : (memref<0x!sycl_item_1_>) -> !llvm.ptr<!sycl_item_1_>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.addrspacecast %[[VAL_5]] : !llvm.ptr<!sycl_item_1_> to !llvm.ptr<!sycl_item_1_, 4>
// CHECK-NEXT:      %[[VAL_7:.*]] = "polygeist.pointer2memref"(%[[VAL_6]]) : (!llvm.ptr<!sycl_item_1_, 4>) -> memref<?x!sycl_item_1_, 4>
// CHECK-NEXT:      %[[VAL_8:.*]] = sycl.call(%[[VAL_7]]) {FunctionName = @get_id, MangledFunctionName = @_ZNK4sycl3_V14itemILi1ELb1EE6get_idEv, TypeName = @item} : (memref<?x!sycl_item_1_, 4>) -> !sycl_id_1_
// CHECK-NEXT:      return %[[VAL_8]] : !sycl_id_1_
// CHECK-NEXT:    }

func.func @sycl_item_get_id(%arg0: !sycl_item_1_) -> !sycl_id_1_ {
  %0 = "sycl.item.get_id"(%arg0) {BaseType = memref<?x!sycl_item_1_, 4>, FunctionName = @get_id, MangledFunctionName = @_ZNK4sycl3_V14itemILi1ELb1EE6get_idEv, TypeName = @item} : (!sycl_item_1_) -> !sycl_id_1_
  return %0 : !sycl_id_1_
}
