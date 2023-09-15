// RUN: polygeist-opt -convert-polygeist-to-llvm -reconcile-unrealized-casts %s | FileCheck %s

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_item_1_ = !sycl.item<[1, false], (!sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>)>
!sycl_item_2_ = !sycl.item<[2, false], (!sycl.item_base<[2, false], (!sycl_range_2_, !sycl_id_2_)>)>
!sycl_item_3_ = !sycl.item<[3, false], (!sycl.item_base<[3, false], (!sycl_range_3_, !sycl_id_3_)>)>

// CHECK-LABEL:   llvm.func @id_default() -> !llvm.struct<(ptr, ptr, ptr)> {
// CHECK-NEXT:      %[[VAL_0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT:      "llvm.intr.memset"(%[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.alloca %[[VAL_5]] x !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.mlir.constant(16 : i64) : i64
// CHECK-NEXT:      "llvm.intr.memset"(%[[VAL_6]], %[[VAL_7]], %[[VAL_8]]) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.alloca %[[VAL_10]] x !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.mlir.constant(24 : i64) : i64
// CHECK-NEXT:      "llvm.intr.memset"(%[[VAL_11]], %[[VAL_12]], %[[VAL_13]]) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      %[[VAL_16:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_15]][0] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      %[[VAL_17:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_16]][1] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      %[[VAL_18:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_17]][2] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      llvm.return %[[VAL_18]] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:    }
func.func @id_default()
    -> (memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>) {
  %id1 = sycl.id.constructor() : () -> memref<1x!sycl_id_1_>
  %id2 = sycl.id.constructor() : () -> memref<1x!sycl_id_2_>
  %id3 = sycl.id.constructor() : () -> memref<1x!sycl_id_3_>
  func.return %id1, %id2, %id3
      : memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>
}

// CHECK-LABEL:   llvm.func @id_index(
// CHECK-SAME:                        %[[VAL_0:.*]]: i64, %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64) -> !llvm.struct<(ptr, ptr, ptr)> {
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_4]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>
// CHECK-NEXT:      llvm.store %[[VAL_0]], %[[VAL_6]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.alloca %[[VAL_7]] x !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.getelementptr inbounds %[[VAL_8]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>
// CHECK-NEXT:      llvm.store %[[VAL_0]], %[[VAL_10]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.getelementptr inbounds %[[VAL_8]][0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>
// CHECK-NEXT:      llvm.store %[[VAL_1]], %[[VAL_12]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.alloca %[[VAL_13]] x !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      %[[VAL_16:.*]] = llvm.getelementptr inbounds %[[VAL_14]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      llvm.store %[[VAL_0]], %[[VAL_16]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_17:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_18:.*]] = llvm.getelementptr inbounds %[[VAL_14]][0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      llvm.store %[[VAL_1]], %[[VAL_18]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_19:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:      %[[VAL_20:.*]] = llvm.getelementptr inbounds %[[VAL_14]][0, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      llvm.store %[[VAL_2]], %[[VAL_20]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_21:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      %[[VAL_22:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_21]][0] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      %[[VAL_23:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_22]][1] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      %[[VAL_24:.*]] = llvm.insertvalue %[[VAL_14]], %[[VAL_23]][2] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      llvm.return %[[VAL_24]] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:    }
func.func @id_index(%arg0: index, %arg1: index, %arg2: index)
    -> (memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>) {
  %id1 = sycl.id.constructor(%arg0)
      : (index) -> memref<1x!sycl_id_1_>
  %id2 = sycl.id.constructor(%arg0, %arg1)
      : (index, index) -> memref<1x!sycl_id_2_>
  %id3 = sycl.id.constructor(%arg0, %arg1, %arg2)
      : (index, index, index) -> memref<1x!sycl_id_3_>
  func.return %id1, %id2, %id3
      : memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>
}

// CHECK-LABEL:   llvm.func @id_range(
// CHECK-SAME:                        %[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: !llvm.ptr, %[[VAL_2:.*]]: !llvm.ptr) -> !llvm.struct<(ptr, ptr, ptr)> {
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_4]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>
// CHECK-NEXT:      llvm.store %[[VAL_7]], %[[VAL_9]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.alloca %[[VAL_10]] x !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_16:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>
// CHECK-NEXT:      %[[VAL_17:.*]] = llvm.load %[[VAL_16]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_18:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      %[[VAL_19:.*]] = llvm.getelementptr inbounds %[[VAL_11]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>
// CHECK-NEXT:      llvm.store %[[VAL_14]], %[[VAL_19]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_20:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_21:.*]] = llvm.getelementptr inbounds %[[VAL_11]][0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>
// CHECK-NEXT:      llvm.store %[[VAL_17]], %[[VAL_21]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_22:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_23:.*]] = llvm.alloca %[[VAL_22]] x !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_24:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      %[[VAL_25:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      %[[VAL_26:.*]] = llvm.load %[[VAL_25]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_27:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_28:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      %[[VAL_29:.*]] = llvm.load %[[VAL_28]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_30:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:      %[[VAL_31:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      %[[VAL_32:.*]] = llvm.load %[[VAL_31]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_33:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      %[[VAL_34:.*]] = llvm.getelementptr inbounds %[[VAL_23]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      llvm.store %[[VAL_26]], %[[VAL_34]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_35:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_36:.*]] = llvm.getelementptr inbounds %[[VAL_23]][0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      llvm.store %[[VAL_29]], %[[VAL_36]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_37:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:      %[[VAL_38:.*]] = llvm.getelementptr inbounds %[[VAL_23]][0, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      llvm.store %[[VAL_32]], %[[VAL_38]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_39:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      %[[VAL_40:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_39]][0] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      %[[VAL_41:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_40]][1] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      %[[VAL_42:.*]] = llvm.insertvalue %[[VAL_23]], %[[VAL_41]][2] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      llvm.return %[[VAL_42]] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:    }
func.func @id_range(%arg0: memref<?x!sycl_range_1_>,
                    %arg1: memref<?x!sycl_range_2_>,
                    %arg2: memref<?x!sycl_range_3_>)
    -> (memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>) {
  %id1 = sycl.id.constructor(%arg0)
      : (memref<?x!sycl_range_1_>) -> memref<1x!sycl_id_1_>
  %id2 = sycl.id.constructor(%arg1)
      : (memref<?x!sycl_range_2_>) -> memref<1x!sycl_id_2_>
  %id3 = sycl.id.constructor(%arg2)
      : (memref<?x!sycl_range_3_>) -> memref<1x!sycl_id_3_>
  func.return %id1, %id2, %id3
      : memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>
}

// CHECK-LABEL:   llvm.func @id_item(
// CHECK-SAME:                       %[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: !llvm.ptr, %[[VAL_2:.*]]: !llvm.ptr) -> !llvm.struct<(ptr, ptr, ptr)> {
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.1.false", (struct<"struct.sycl::_V1::detail::ItemBase.1.false", (struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>, struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>)>)>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_4]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>
// CHECK-NEXT:      llvm.store %[[VAL_7]], %[[VAL_9]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.alloca %[[VAL_10]] x !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.2.false", (struct<"struct.sycl::_V1::detail::ItemBase.2.false", (struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>)>)>
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_16:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 1, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.2.false", (struct<"struct.sycl::_V1::detail::ItemBase.2.false", (struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>)>)>
// CHECK-NEXT:      %[[VAL_17:.*]] = llvm.load %[[VAL_16]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_18:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      %[[VAL_19:.*]] = llvm.getelementptr inbounds %[[VAL_11]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>
// CHECK-NEXT:      llvm.store %[[VAL_14]], %[[VAL_19]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_20:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_21:.*]] = llvm.getelementptr inbounds %[[VAL_11]][0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>
// CHECK-NEXT:      llvm.store %[[VAL_17]], %[[VAL_21]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_22:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_23:.*]] = llvm.alloca %[[VAL_22]] x !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_24:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      %[[VAL_25:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.3.false", (struct<"struct.sycl::_V1::detail::ItemBase.3.false", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>)>
// CHECK-NEXT:      %[[VAL_26:.*]] = llvm.load %[[VAL_25]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_27:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_28:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0, 1, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.3.false", (struct<"struct.sycl::_V1::detail::ItemBase.3.false", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>)>
// CHECK-NEXT:      %[[VAL_29:.*]] = llvm.load %[[VAL_28]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_30:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:      %[[VAL_31:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0, 1, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.3.false", (struct<"struct.sycl::_V1::detail::ItemBase.3.false", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>)>
// CHECK-NEXT:      %[[VAL_32:.*]] = llvm.load %[[VAL_31]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_33:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      %[[VAL_34:.*]] = llvm.getelementptr inbounds %[[VAL_23]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      llvm.store %[[VAL_26]], %[[VAL_34]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_35:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_36:.*]] = llvm.getelementptr inbounds %[[VAL_23]][0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      llvm.store %[[VAL_29]], %[[VAL_36]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_37:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:      %[[VAL_38:.*]] = llvm.getelementptr inbounds %[[VAL_23]][0, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      llvm.store %[[VAL_32]], %[[VAL_38]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_39:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      %[[VAL_40:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_39]][0] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      %[[VAL_41:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_40]][1] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      %[[VAL_42:.*]] = llvm.insertvalue %[[VAL_23]], %[[VAL_41]][2] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      llvm.return %[[VAL_42]] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:    }
func.func @id_item(%arg0: memref<?x!sycl_item_1_>,
                   %arg1: memref<?x!sycl_item_2_>,
                   %arg2: memref<?x!sycl_item_3_>)
    -> (memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>) {
  %id1 = sycl.id.constructor(%arg0)
      : (memref<?x!sycl_item_1_>) -> memref<1x!sycl_id_1_>
  %id2 = sycl.id.constructor(%arg1)
      : (memref<?x!sycl_item_2_>) -> memref<1x!sycl_id_2_>
  %id3 = sycl.id.constructor(%arg2)
      : (memref<?x!sycl_item_3_>) -> memref<1x!sycl_id_3_>
  func.return %id1, %id2, %id3
      : memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>
}

// CHECK-LABEL:   llvm.func @id_id(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: !llvm.ptr, %[[VAL_2:.*]]: !llvm.ptr) -> !llvm.struct<(ptr, ptr, ptr)> {
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT:      "llvm.intr.memcpy"(%[[VAL_4]], %[[VAL_0]], %[[VAL_5]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.alloca %[[VAL_7]] x !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.mlir.constant(16 : i64) : i64
// CHECK-NEXT:      "llvm.intr.memcpy"(%[[VAL_8]], %[[VAL_1]], %[[VAL_9]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.alloca %[[VAL_11]] x !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.mlir.constant(24 : i64) : i64
// CHECK-NEXT:      "llvm.intr.memcpy"(%[[VAL_12]], %[[VAL_2]], %[[VAL_13]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      %[[VAL_16:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_15]][0] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      %[[VAL_17:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_16]][1] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      %[[VAL_18:.*]] = llvm.insertvalue %[[VAL_12]], %[[VAL_17]][2] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:      llvm.return %[[VAL_18]] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK-NEXT:    }
func.func @id_id(%arg0: memref<?x!sycl_id_1_>,
                 %arg1: memref<?x!sycl_id_2_>,
                 %arg2: memref<?x!sycl_id_3_>)
    -> (memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>) {
  %id1 = sycl.id.constructor(%arg0)
      : (memref<?x!sycl_id_1_>) -> memref<1x!sycl_id_1_>
  %id2 = sycl.id.constructor(%arg1)
      : (memref<?x!sycl_id_2_>) -> memref<1x!sycl_id_2_>
  %id3 = sycl.id.constructor(%arg2)
      : (memref<?x!sycl_id_3_>) -> memref<1x!sycl_id_3_>
  func.return %id1, %id2, %id3
      : memref<1x!sycl_id_1_>, memref<1x!sycl_id_2_>, memref<1x!sycl_id_3_>
}
