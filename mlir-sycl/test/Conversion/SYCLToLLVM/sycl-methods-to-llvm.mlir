// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// sycl.range.get with scalar result type
//===----------------------------------------------------------------------===//

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, %[[VAL_1]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      llvm.return %[[VAL_3]] : i64
func.func @test(%range: memref<?x!sycl_range_3_>, %idx: i32) -> i64 {
  %0 = sycl.range.get %range[%idx] : (memref<?x!sycl_range_3_>, i32) -> i64
  return %0 : i64
}

// -----

//===----------------------------------------------------------------------===//
// sycl.range.get with reference result type
//===----------------------------------------------------------------------===//

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> !llvm.ptr {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, %[[VAL_1]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.3", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.ptr
func.func @test(%range: memref<?x!sycl_range_3_>, %idx: i32) -> memref<?xi64> {
  %0 = sycl.range.get %range[%idx] : (memref<?x!sycl_range_3_>, i32) -> memref<?xi64>
  return %0 : memref<?xi64>
}

// -----

//===----------------------------------------------------------------------===//
// sycl.range.size
//===----------------------------------------------------------------------===//

!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>

// CHECK-LABEL:   llvm.func @test_1(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mul %[[VAL_3]], %[[VAL_1]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_4]] : i64
func.func @test_1(%range: memref<?x!sycl_range_1_>) -> i64 {
  %0 = sycl.range.size(%range) : (memref<?x!sycl_range_1_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mul %[[VAL_3]], %[[VAL_1]]  : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_4]], %[[VAL_6]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_7]] : i64
func.func @test_2(%range: memref<?x!sycl_range_2_>) -> i64 {
  %0 = sycl.range.size(%range) : (memref<?x!sycl_range_2_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mul %[[VAL_3]], %[[VAL_1]]  : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_4]], %[[VAL_6]]  : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.mul %[[VAL_7]], %[[VAL_9]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_10]] : i64
func.func @test_3(%range: memref<?x!sycl_range_3_>) -> i64 {
  %0 = sycl.range.size(%range) : (memref<?x!sycl_range_3_>) -> i64
  return %0 : i64
}

// -----

//===----------------------------------------------------------------------===//
// sycl.id.get with scalar result type
//===----------------------------------------------------------------------===//

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, %[[VAL_1]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      llvm.return %[[VAL_3]] : i64
func.func @test(%id: memref<?x!sycl_id_3_>, %idx: i32) -> i64 {
  %0 = sycl.id.get %id[%idx] : (memref<?x!sycl_id_3_>, i32) -> i64
  return %0 : i64
}

// -----

//===----------------------------------------------------------------------===//
// sycl.id.get with scalar result type and no argument
//===----------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, %[[VAL_1]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      llvm.return %[[VAL_3]] : i64
func.func @test(%id: memref<?x!sycl_id_1_>) -> i64 {
  %0 = sycl.id.get %id[] : (memref<?x!sycl_id_1_>) -> i64
  return %0 : i64
}

// -----

//===----------------------------------------------------------------------===//
// sycl.id.get with reference result type
//===----------------------------------------------------------------------===//

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> !llvm.ptr {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, %[[VAL_1]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.ptr
func.func @test(%id: memref<?x!sycl_id_3_>, %idx: i32) -> memref<?xi64> {
  %0 = sycl.id.get %id[%idx] : (memref<?x!sycl_id_3_>, i32) -> memref<?xi64>
  return %0 : memref<?xi64>
}

// -----

//===----------------------------------------------------------------------===//
// sycl.accessor.get_pointer
//===----------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.ptr<1> {
// CHECK-DAG:       %[[VAL_1:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !sycl_accessor_1_i32_rw_gb
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !sycl_accessor_1_i32_rw_gb
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_4]], %[[VAL_2]]  : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.add %[[VAL_7]], %[[VAL_6]]  : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.sub %[[VAL_1]], %[[VAL_8]]  : i64
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::accessor.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.load %[[VAL_10]] : !llvm.ptr -> !llvm.ptr<1>
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.getelementptr inbounds %[[VAL_11]]{{\[}}%[[VAL_9]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
// CHECK-NEXT:      llvm.return %[[VAL_12]] : !llvm.ptr<1>
func.func @test(%acc: memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?xi32, 1> {
  %0 = sycl.accessor.get_pointer(%acc) : (memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?xi32, 1>
  return %0 : memref<?xi32, 1>
}

// -----

//===----------------------------------------------------------------------===//
// sycl.accessor.get_range
//===----------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.struct<"class.sycl::_V1::range.1", {{.*}}> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::accessor.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
func.func @test(%acc: memref<?x!sycl_accessor_1_i32_rw_gb>) -> !sycl_range_1_ {
  %0 = sycl.accessor.get_range(%acc) : (memref<?x!sycl_accessor_1_i32_rw_gb>) -> !sycl_range_1_
  return %0 : !sycl_range_1_
}

// -----

//===----------------------------------------------------------------------===//
// sycl.accessor.size
//===----------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::accessor.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mul %[[VAL_3]], %[[VAL_1]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_4]] : i64
// CHECK-NEXT:    }
func.func @test(%acc: memref<?x!sycl_accessor_1_i32_rw_gb>) -> i64 {
  %0 = sycl.accessor.size(%acc) : (memref<?x!sycl_accessor_1_i32_rw_gb>) -> i64
  return %0 : i64
}

// -----

//===----------------------------------------------------------------------===//
// sycl.accessor.subscript with scalar offset and 1D accessor
//===----------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:.*]]: i64) -> !llvm.ptr<1> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::accessor.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> !llvm.ptr<1>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_3]]{{\[}}%[[VAL_1]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
// CHECK-NEXT:      llvm.return %[[VAL_4]] : !llvm.ptr<1>
func.func @test(%acc: memref<?x!sycl_accessor_1_i32_rw_gb>, %idx: i64) -> memref<?xi32, 1> {
  %0 = sycl.accessor.subscript %acc[%idx] : (memref<?x!sycl_accessor_1_i32_rw_gb>, i64) -> memref<?xi32, 1>
  return %0 : memref<?xi32, 1>
}

// -----

//===----------------------------------------------------------------------===//
// sycl.accessor.subscript with scalar offset and ND accessor
//===----------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_accessor_impl_device_2_ = !sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>
!sycl_accessor_2_i32_rw_gb = !sycl.accessor<[2, i32, read_write, global_buffer], (!sycl_accessor_impl_device_2_, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_accessor_impl_device_3_ = !sycl.accessor_impl_device<[3], (!sycl_id_3_, !sycl_range_3_, !sycl_range_3_)>
!sycl_accessor_3_i32_rw_gb = !sycl.accessor<[3, i32, read_write, global_buffer], (!sycl_accessor_impl_device_3_, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_accessor_subscript_2_ = !sycl.accessor_subscript<[2], (!sycl_id_2_, !sycl_accessor_2_i32_rw_gb)>
!sycl_accessor_subscript_3_ = !sycl.accessor_subscript<[3], (!sycl_id_3_, !sycl_accessor_3_i32_rw_gb)>

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                      %[[VAL_1:.*]]: i64) -> !llvm.struct<"class.sycl::_V1::detail::accessor_common.AccessorSubscript.2", {{.*}}> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.undef : !llvm.struct<"class.sycl::_V1::detail::accessor_common.AccessorSubscript.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_2]][0, 0, 0, 0] : !llvm.struct<"class.sycl::_V1::detail::accessor_common.AccessorSubscript.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.load %[[VAL_0]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::accessor.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_3]][1] : !llvm.struct<"class.sycl::_V1::detail::accessor_common.AccessorSubscript.2", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_6]] : !llvm.struct<"class.sycl::_V1::detail::accessor_common.AccessorSubscript.2", {{.*}}>
func.func @test_2(%acc: memref<?x!sycl_accessor_2_i32_rw_gb>, %idx: i64) -> !sycl_accessor_subscript_2_ {
  %0 = sycl.accessor.subscript %acc[%idx] : (memref<?x!sycl_accessor_2_i32_rw_gb>, i64) -> !sycl_accessor_subscript_2_
  return %0 : !sycl_accessor_subscript_2_
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                      %[[VAL_1:.*]]: i64) -> !llvm.struct<"class.sycl::_V1::detail::accessor_common.AccessorSubscript.3", {{.*}}> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.undef : !llvm.struct<"class.sycl::_V1::detail::accessor_common.AccessorSubscript.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_2]][0, 0, 0, 0] : !llvm.struct<"class.sycl::_V1::detail::accessor_common.AccessorSubscript.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_3]][0, 0, 0, 1] : !llvm.struct<"class.sycl::_V1::detail::accessor_common.AccessorSubscript.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_0]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::accessor.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_5]][1] : !llvm.struct<"class.sycl::_V1::detail::accessor_common.AccessorSubscript.3", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_7]] : !llvm.struct<"class.sycl::_V1::detail::accessor_common.AccessorSubscript.3", {{.*}}>
func.func @test_3(%acc: memref<?x!sycl_accessor_3_i32_rw_gb>, %idx: i64) -> !sycl_accessor_subscript_3_ {
  %0 = sycl.accessor.subscript %acc[%idx] : (memref<?x!sycl_accessor_3_i32_rw_gb>, i64) -> !sycl_accessor_subscript_3_
  return %0 : !sycl_accessor_subscript_3_
}

// -----

//===----------------------------------------------------------------------===//
// sycl.accessor.subscript with id offset
//===----------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_accessor_impl_device_2_ = !sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>
!sycl_accessor_2_i32_rw_gb = !sycl.accessor<[2, i32, read_write, global_buffer], (!sycl_accessor_impl_device_2_, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_accessor_impl_device_3_ = !sycl.accessor_impl_device<[3], (!sycl_id_3_, !sycl_range_3_, !sycl_range_3_)>
!sycl_accessor_3_i32_rw_gb = !sycl.accessor<[3, i32, read_write, global_buffer], (!sycl_accessor_impl_device_3_, !llvm.struct<(ptr<i32, 1>)>)>
!my_struct = !llvm.struct<(i32, f32)>
!sycl_accessor_1_struct_rw_gb = !sycl.accessor<[1, !my_struct, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<!my_struct, 1>)>)>

// CHECK-LABEL:   llvm.func @test_1(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                      %[[VAL_1:.*]]: !llvm.ptr) -> !llvm.ptr<1> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::accessor.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_4]], %[[VAL_2]]  : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.add %[[VAL_7]], %[[VAL_6]]  : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::accessor.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr -> !llvm.ptr<1>
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_10]]{{\[}}%[[VAL_8]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
// CHECK-NEXT:      llvm.return %[[VAL_11]] : !llvm.ptr<1>
func.func @test_1(%acc: memref<?x!sycl_accessor_1_i32_rw_gb>, %idx: memref<?x!sycl_id_1_>) -> memref<?xi32, 1> {
  %0 = sycl.accessor.subscript %acc[%idx] : (memref<?x!sycl_accessor_1_i32_rw_gb>, memref<?x!sycl_id_1_>) -> memref<?xi32, 1>
  return %0 : memref<?xi32, 1>
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                      %[[VAL_1:.*]]: !llvm.ptr) -> !llvm.ptr<1> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::accessor.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_4]], %[[VAL_2]]  : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.add %[[VAL_7]], %[[VAL_6]]  : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::accessor.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.mul %[[VAL_8]], %[[VAL_10]]  : i64
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.add %[[VAL_13]], %[[VAL_12]]  : i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::accessor.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_16:.*]] = llvm.load %[[VAL_15]] : !llvm.ptr -> !llvm.ptr<1>
// CHECK-NEXT:      %[[VAL_17:.*]] = llvm.getelementptr inbounds %[[VAL_16]]{{\[}}%[[VAL_14]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
// CHECK-NEXT:      llvm.return %[[VAL_17]] : !llvm.ptr<1>
func.func @test_2(%acc: memref<?x!sycl_accessor_2_i32_rw_gb>, %idx: memref<?x!sycl_id_2_>) -> memref<?xi32, 1> {
  %0 = sycl.accessor.subscript %acc[%idx] : (memref<?x!sycl_accessor_2_i32_rw_gb>, memref<?x!sycl_id_2_>) -> memref<?xi32, 1>
  return %0 : memref<?xi32, 1>
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                      %[[VAL_1:.*]]: !llvm.ptr) -> !llvm.ptr<1> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::accessor.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_4]], %[[VAL_2]]  : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.add %[[VAL_7]], %[[VAL_6]]  : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::accessor.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.mul %[[VAL_8]], %[[VAL_10]]  : i64
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.add %[[VAL_13]], %[[VAL_12]]  : i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::accessor.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_16:.*]] = llvm.load %[[VAL_15]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_17:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_18:.*]] = llvm.load %[[VAL_17]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_19:.*]] = llvm.mul %[[VAL_14]], %[[VAL_16]]  : i64
// CHECK-NEXT:      %[[VAL_20:.*]] = llvm.add %[[VAL_19]], %[[VAL_18]]  : i64
// CHECK-NEXT:      %[[VAL_21:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::accessor.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_22:.*]] = llvm.load %[[VAL_21]] : !llvm.ptr -> !llvm.ptr<1>
// CHECK-NEXT:      %[[VAL_23:.*]] = llvm.getelementptr inbounds %[[VAL_22]]{{\[}}%[[VAL_20]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
// CHECK-NEXT:      llvm.return %[[VAL_23]] : !llvm.ptr<1>
func.func @test_3(%acc: memref<?x!sycl_accessor_3_i32_rw_gb>, %idx: memref<?x!sycl_id_3_>) -> memref<?xi32, 1> {
  %0 = sycl.accessor.subscript %acc[%idx] : (memref<?x!sycl_accessor_3_i32_rw_gb>, memref<?x!sycl_id_3_>) -> memref<?xi32, 1>
  return %0 : memref<?xi32, 1>
}

// CHECK-LABEL:   llvm.func @test_struct(
// CHECK-SAME:                           %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                           %[[VAL_1:.*]]: !llvm.ptr) -> !llvm.ptr<1> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::accessor.1.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_4]], %[[VAL_2]]  : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.add %[[VAL_7]], %[[VAL_6]]  : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::accessor.1.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr -> !llvm.ptr<1>
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_10]]{{\[}}%[[VAL_8]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, !llvm.struct<(i32, f32)>
// CHECK-NEXT:      llvm.return %[[VAL_11]] : !llvm.ptr<1>
func.func @test_struct(%acc: memref<?x!sycl_accessor_1_struct_rw_gb>, %idx: memref<?x!sycl_id_1_>) -> !llvm.ptr<1> {
  %0 = sycl.accessor.subscript %acc[%idx] : (memref<?x!sycl_accessor_1_struct_rw_gb>, memref<?x!sycl_id_1_>) -> !llvm.ptr<1>
  return %0 : !llvm.ptr<1>
}

// -----


//===----------------------------------------------------------------------===//
// sycl.accessor.subscript with id offset and atomic return type
//===----------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_ato_gb = !sycl.accessor<[1, i32, atomic, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_atomic_i32_glo = !sycl.atomic<[i32, global], (memref<?xi32, 1>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:.*]]: !llvm.ptr) -> !llvm.struct<"class.sycl::_V1::atomic", {{.*}}> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.undef : !llvm.struct<"class.sycl::_V1::atomic", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::accessor.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.mul %[[VAL_5]], %[[VAL_3]]  : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.add %[[VAL_8]], %[[VAL_7]]  : i64
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::accessor.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.load %[[VAL_10]] : !llvm.ptr -> !llvm.ptr<1>
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.getelementptr inbounds %[[VAL_11]]{{\[}}%[[VAL_9]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_12]], %[[VAL_2]][0] : !llvm.struct<"class.sycl::_V1::atomic", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_13]] : !llvm.struct<"class.sycl::_V1::atomic", {{.*}}>
func.func @test(%acc: memref<?x!sycl_accessor_1_i32_ato_gb>, %idx: memref<?x!sycl_id_1_>) -> !sycl_atomic_i32_glo {
  %0 = sycl.accessor.subscript %acc[%idx] : (memref<?x!sycl_accessor_1_i32_ato_gb>, memref<?x!sycl_id_1_>) -> !sycl_atomic_i32_glo
  return %0 : !sycl_atomic_i32_glo
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_nd_range_1_ = !sycl.nd_range<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.struct<"class.sycl::_V1::range.1", {{.*}}> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
func.func @test(%nd: memref<?x!sycl_nd_range_1_>) -> !sycl_range_1_ {
  %0 = sycl.nd_range.get_global_range(%nd) : (memref<?x!sycl_nd_range_1_>) -> !sycl_range_1_
  return %0 : !sycl_range_1_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_nd_range_1_ = !sycl.nd_range<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.struct<"class.sycl::_V1::range.1", {{.*}}> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
func.func @test(%nd: memref<?x!sycl_nd_range_1_>) -> !sycl_range_1_ {
  %0 = sycl.nd_range.get_local_range(%nd) : (memref<?x!sycl_nd_range_1_>) -> !sycl_range_1_
  return %0 : !sycl_range_1_
}

// -----

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_nd_range_3_ = !sycl.nd_range<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.struct<"class.sycl::_V1::range.3", {{.*}}> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::range.3", {{.*}}> : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.udiv %[[VAL_4]], %[[VAL_6]]  : i64
// CHECK-NEXT:       %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.3", {{.*}}>
// CHECK-NEXT:      llvm.store %[[VAL_7]], %[[VAL_8]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.udiv %[[VAL_10]], %[[VAL_12]]  : i64
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.3", {{.*}}>
// CHECK-NEXT:      llvm.store %[[VAL_13]], %[[VAL_14]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_16:.*]] = llvm.load %[[VAL_15]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_17:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_18:.*]] = llvm.load %[[VAL_17]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_19:.*]] = llvm.udiv %[[VAL_16]], %[[VAL_18]]  : i64
// CHECK-NEXT:      %[[VAL_20:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.3", {{.*}}>
// CHECK-NEXT:      llvm.store %[[VAL_19]], %[[VAL_20]] : i64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_21:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::range.3", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_21]] : !llvm.struct<"class.sycl::_V1::range.3", {{.*}}>
func.func @test(%nd: memref<?x!sycl_nd_range_3_>) -> !sycl_range_3_ {
  %0 = sycl.nd_range.get_group_range(%nd) : (memref<?x!sycl_nd_range_3_>) -> !sycl_range_3_
  return %0 : !sycl_range_3_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, false], (!sycl_item_base_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.struct<"class.sycl::_V1::id.1", {{.*}}> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.1.false", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK-NEXT:    }
func.func @test(%item: memref<?x!sycl_item_1_>) -> !sycl_id_1_ {
  %0 = sycl.item.get_id(%item) : (memref<?x!sycl_item_1_>) -> !sycl_id_1_
  return %0 : !sycl_id_1_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, false], (!sycl_item_base_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, %[[VAL_1]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.1.false", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:    }
func.func @test(%item: memref<?x!sycl_item_1_>, %i: i32) -> i64 {
  %0 = sycl.item.get_id(%item, %i) : (memref<?x!sycl_item_1_>, i32) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, false], (!sycl_item_base_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, %[[VAL_1]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.1.false", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:    }
func.func @test(%item: memref<?x!sycl_item_1_>) -> i64 {
  %0 = sycl.item.get_id(%item) : (memref<?x!sycl_item_1_>) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, false], (!sycl_item_base_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.struct<"class.sycl::_V1::range.1", {{.*}}> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.1.false", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
// CHECK-NEXT:    }
func.func @test(%item: memref<?x!sycl_item_1_>) -> !sycl_range_1_ {
  %0 = sycl.item.get_range(%item) : (memref<?x!sycl_item_1_>) -> !sycl_range_1_
  return %0 : !sycl_range_1_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, false], (!sycl_item_base_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, %[[VAL_1]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.1.false", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:    }
func.func @test(%item: memref<?x!sycl_item_1_>, %i: i32) -> i64 {
  %0 = sycl.item.get_range(%item, %i) : (memref<?x!sycl_item_1_>, i32) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, false], (!sycl_item_base_1_)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_item_base_2_ = !sycl.item_base<[2, false], (!sycl_range_2_, !sycl_id_2_)>
!sycl_item_2_ = !sycl.item<[2, false], (!sycl_item_base_2_)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_item_base_3_ = !sycl.item_base<[3, false], (!sycl_range_3_, !sycl_id_3_)>
!sycl_item_3_ = !sycl.item<[3, false], (!sycl_item_base_3_)>

// CHECK-LABEL:   llvm.func @test_1(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.1.false", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i64
// CHECK-NEXT:      llvm.return %[[VAL_2]] : i64
// CHECK-NEXT:    }
func.func @test_1(%item: memref<?x!sycl_item_1_>) -> i64 {
  %0 = sycl.item.get_linear_id(%item) : (memref<?x!sycl_item_1_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.2.false", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.2.false", {{.*}}>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.2.false", {{.*}}>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.add %[[VAL_5]], %[[VAL_7]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_8]] : i64
// CHECK-NEXT:    }
func.func @test_2(%item: memref<?x!sycl_item_2_>) -> i64 {
  %0 = sycl.item.get_linear_id(%item) : (memref<?x!sycl_item_2_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.3.false", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.3.false", {{.*}}>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.3.false", {{.*}}>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.mul %[[VAL_5]], %[[VAL_7]]  : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.3.false", {{.*}}>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.mul %[[VAL_10]], %[[VAL_7]]  : i64
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.add %[[VAL_8]], %[[VAL_11]]  : i64
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.3.false", {{.*}}>
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.add %[[VAL_12]], %[[VAL_14]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_15]] : i64
// CHECK-NEXT:    }
func.func @test_3(%item: memref<?x!sycl_item_3_>) -> i64 {
  %0 = sycl.item.get_linear_id(%item) : (memref<?x!sycl_item_3_>) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_item_base_2_ = !sycl.item_base<[2, true], (!sycl_range_2_, !sycl_id_2_, !sycl_id_2_)>
!sycl_item_2_ = !sycl.item<[2, true], (!sycl_item_base_2_)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_item_base_3_ = !sycl.item_base<[3, true], (!sycl_range_3_, !sycl_id_3_, !sycl_id_3_)>
!sycl_item_3_ = !sycl.item<[3, true], (!sycl_item_base_3_)>

// CHECK-LABEL:   llvm.func @test_1(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.1.true", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.1.true", {{.*}}>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.sub %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_5]] : i64
// CHECK-NEXT:    }
func.func @test_1(%item: memref<?x!sycl_item_1_>) -> i64 {
  %0 = sycl.item.get_linear_id(%item) : (memref<?x!sycl_item_1_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.2.true", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.2.true", {{.*}}>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.sub %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.2.true", {{.*}}>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.mul %[[VAL_5]], %[[VAL_7]]  : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.2.true", {{.*}}>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.2.true", {{.*}}>
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.sub %[[VAL_10]], %[[VAL_12]]  : i64
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.add %[[VAL_8]], %[[VAL_13]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_14]] : i64
// CHECK-NEXT:    }
func.func @test_2(%item: memref<?x!sycl_item_2_>) -> i64 {
  %0 = sycl.item.get_linear_id(%item) : (memref<?x!sycl_item_2_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.3.true", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.3.true", {{.*}}>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.sub %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.3.true", {{.*}}>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.mul %[[VAL_5]], %[[VAL_7]]  : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.3.true", {{.*}}>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.mul %[[VAL_8]], %[[VAL_10]]  : i64
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.3.true", {{.*}}>
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.load %[[VAL_12]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.3.true", {{.*}}>
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.load %[[VAL_14]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_16:.*]] = llvm.sub %[[VAL_13]], %[[VAL_15]]  : i64
// CHECK-NEXT:      %[[VAL_17:.*]] = llvm.mul %[[VAL_16]], %[[VAL_10]]  : i64
// CHECK-NEXT:      %[[VAL_18:.*]] = llvm.add %[[VAL_11]], %[[VAL_17]]  : i64
// CHECK-NEXT:      %[[VAL_19:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.3.true", {{.*}}>
// CHECK-NEXT:      %[[VAL_20:.*]] = llvm.load %[[VAL_19]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_21:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::item.3.true", {{.*}}>
// CHECK-NEXT:      %[[VAL_22:.*]] = llvm.load %[[VAL_21]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_23:.*]] = llvm.sub %[[VAL_20]], %[[VAL_22]]  : i64
// CHECK-NEXT:      %[[VAL_24:.*]] = llvm.add %[[VAL_18]], %[[VAL_23]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_24]] : i64
// CHECK-NEXT:    }
func.func @test_3(%item: memref<?x!sycl_item_3_>) -> i64 {
  %0 = sycl.item.get_linear_id(%item) : (memref<?x!sycl_item_3_>) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1 = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.struct<"class.sycl::_V1::id.1", {{.*}}> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK-NEXT:    }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>) -> !sycl_id_1_ {
  %0 = sycl.nd_item.get_global_id(%nd) : (memref<?x!sycl_nd_item_1_>) -> !sycl_id_1_
  return %0 : !sycl_id_1_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1 = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1, 0, 0, %[[VAL_1]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:    }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>, %i: i32) -> i64 {
  %0 = sycl.nd_item.get_global_id(%nd, %i) : (memref<?x!sycl_nd_item_1_>, i32) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_item_base_2_ = !sycl.item_base<[2, true], (!sycl_range_2_, !sycl_id_2_, !sycl_id_2_)>
!sycl_item_2_ = !sycl.item<[2, true], (!sycl_item_base_2_)>
!sycl_item_base_2_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_2_1_ = !sycl.item<[1, false], (!sycl_item_base_2_1_)>
!sycl_group_2_ = !sycl.group<[2], (!sycl_range_2_, !sycl_range_2_, !sycl_range_2_, !sycl_id_2_)>
!sycl_nd_item_2_ = !sycl.nd_item<[2], (!sycl_item_2_, !sycl_item_2_1_, !sycl_group_2_)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_item_base_3_ = !sycl.item_base<[3, true], (!sycl_range_3_, !sycl_id_3_, !sycl_id_3_)>
!sycl_item_3_ = !sycl.item<[3, true], (!sycl_item_base_3_)>
!sycl_item_base_3_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_3_1_ = !sycl.item<[1, false], (!sycl_item_base_3_1_)>
!sycl_group_3_ = !sycl.group<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>
!sycl_nd_item_3_ = !sycl.nd_item<[3], (!sycl_item_3_, !sycl_item_3_1_, !sycl_group_3_)>

// CHECK-LABEL:   llvm.func @test_1(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i64
// CHECK-NEXT:      llvm.return %[[VAL_2]] : i64
// CHECK-NEXT:    }
func.func @test_1(%nd: memref<?x!sycl_nd_item_1_>) -> i64 {
  %0 = sycl.nd_item.get_global_linear_id(%nd) : (memref<?x!sycl_nd_item_1_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.add %[[VAL_5]], %[[VAL_7]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_8]] : i64
// CHECK-NEXT:    }
func.func @test_2(%nd: memref<?x!sycl_nd_item_2_>) -> i64 {
  %0 = sycl.nd_item.get_global_linear_id(%nd)  : (memref<?x!sycl_nd_item_2_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.mul %[[VAL_5]], %[[VAL_7]]  : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.mul %[[VAL_10]], %[[VAL_7]]  : i64
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.add %[[VAL_8]], %[[VAL_11]]  : i64
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.add %[[VAL_12]], %[[VAL_14]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_15]] : i64
// CHECK-NEXT:    }
func.func @test_3(%nd: memref<?x!sycl_nd_item_3_>) -> i64 {
  %0 = sycl.nd_item.get_global_linear_id(%nd) : (memref<?x!sycl_nd_item_3_>) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1 = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.struct<"class.sycl::_V1::id.1", {{.*}}> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK-NEXT:    }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>) -> !sycl_id_1_ {
  %0 = sycl.nd_item.get_local_id(%nd) : (memref<?x!sycl_nd_item_1_>) -> !sycl_id_1_
  return %0 : !sycl_id_1_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1 = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 1, 0, 0, %[[VAL_1]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:    }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>, %i: i32) -> i64 {
  %0 = sycl.nd_item.get_local_id(%nd, %i) : (memref<?x!sycl_nd_item_1_>, i32) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_item_base_2_ = !sycl.item_base<[2, true], (!sycl_range_2_, !sycl_id_2_, !sycl_id_2_)>
!sycl_item_2_ = !sycl.item<[2, true], (!sycl_item_base_2_)>
!sycl_item_base_2_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_2_1_ = !sycl.item<[1, false], (!sycl_item_base_2_1_)>
!sycl_group_2_ = !sycl.group<[2], (!sycl_range_2_, !sycl_range_2_, !sycl_range_2_, !sycl_id_2_)>
!sycl_nd_item_2_ = !sycl.nd_item<[2], (!sycl_item_2_, !sycl_item_2_1_, !sycl_group_2_)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_item_base_3_ = !sycl.item_base<[3, true], (!sycl_range_3_, !sycl_id_3_, !sycl_id_3_)>
!sycl_item_3_ = !sycl.item<[3, true], (!sycl_item_base_3_)>
!sycl_item_base_3_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_3_1_ = !sycl.item<[1, false], (!sycl_item_base_3_1_)>
!sycl_group_3_ = !sycl.group<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>
!sycl_nd_item_3_ = !sycl.nd_item<[3], (!sycl_item_3_, !sycl_item_3_1_, !sycl_group_3_)>

// CHECK-LABEL:   llvm.func @test_1(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i64
// CHECK-NEXT:      llvm.return %[[VAL_2]] : i64
// CHECK-NEXT:    }
func.func @test_1(%nd: memref<?x!sycl_nd_item_1_>) -> i64 {
  %0 = sycl.nd_item.get_local_linear_id(%nd) : (memref<?x!sycl_nd_item_1_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 1, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.add %[[VAL_5]], %[[VAL_7]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_8]] : i64
// CHECK-NEXT:    }
func.func @test_2(%nd: memref<?x!sycl_nd_item_2_>) -> i64 {
  %0 = sycl.nd_item.get_local_linear_id(%nd) : (memref<?x!sycl_nd_item_2_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.mul %[[VAL_5]], %[[VAL_7]]  : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 1, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.mul %[[VAL_10]], %[[VAL_7]]  : i64
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.add %[[VAL_8]], %[[VAL_11]]  : i64
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 1, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.add %[[VAL_12]], %[[VAL_14]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_15]] : i64
// CHECK-NEXT:    }
func.func @test_3(%nd: memref<?x!sycl_nd_item_3_>) -> i64 {
  %0 = sycl.nd_item.get_local_linear_id(%nd) : (memref<?x!sycl_nd_item_3_>) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_item_base_2_ = !sycl.item_base<[2, true], (!sycl_range_2_, !sycl_id_2_, !sycl_id_2_)>
!sycl_item_2_ = !sycl.item<[2, true], (!sycl_item_base_2_)>
!sycl_item_base_2_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_2_1_ = !sycl.item<[1, false], (!sycl_item_base_2_1_)>
!sycl_group_2_ = !sycl.group<[2], (!sycl_range_2_, !sycl_range_2_, !sycl_range_2_, !sycl_id_2_)>
!sycl_nd_item_2_ = !sycl.nd_item<[2], (!sycl_item_2_, !sycl_item_2_1_, !sycl_group_2_)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_item_base_3_ = !sycl.item_base<[3, true], (!sycl_range_3_, !sycl_id_3_, !sycl_id_3_)>
!sycl_item_3_ = !sycl.item<[3, true], (!sycl_item_base_3_)>
!sycl_item_base_3_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_3_1_ = !sycl.item<[1, false], (!sycl_item_base_3_1_)>
!sycl_group_3_ = !sycl.group<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>
!sycl_nd_item_3_ = !sycl.nd_item<[3], (!sycl_item_3_, !sycl_item_3_1_, !sycl_group_3_)>

// CHECK-LABEL:   llvm.func @test_1(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.1", {{.*}}>
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 3, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.1", {{.*}}>
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK:           llvm.return %[[VAL_4]] : i64
// CHECK:    }
func.func @test_1(%nd: memref<?x!sycl_nd_item_1_>) -> i64 {
  %0 = sycl.nd_item.get_group_linear_id(%nd) : (memref<?x!sycl_nd_item_1_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.2", {{.*}}>
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 3, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.2", {{.*}}>
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 2, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.2", {{.*}}>
// CHECK:           %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_7:.*]] = llvm.mul %[[VAL_4]], %[[VAL_6]]  : i64
// CHECK:           %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 3, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.2", {{.*}}>
// CHECK:           %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_10:.*]] = llvm.add %[[VAL_7]], %[[VAL_9]]  : i64
// CHECK:           llvm.return %[[VAL_10]] : i64
// CHECK:    }
func.func @test_2(%nd: memref<?x!sycl_nd_item_2_>) -> i64 {
  %0 = sycl.nd_item.get_group_linear_id(%nd) : (memref<?x!sycl_nd_item_2_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.3", {{.*}}>
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 3, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 2, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK:           %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_7:.*]] = llvm.mul %[[VAL_4]], %[[VAL_6]]  : i64
// CHECK:           %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 2, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK:           %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_10:.*]] = llvm.mul %[[VAL_7]], %[[VAL_9]]  : i64
// CHECK:           %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 3, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK:           %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_13:.*]] = llvm.mul %[[VAL_12]], %[[VAL_9]]  : i64
// CHECK:           %[[VAL_14:.*]] = llvm.add %[[VAL_10]], %[[VAL_13]]  : i64
// CHECK:           %[[VAL_15:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 3, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK:           %[[VAL_16:.*]] = llvm.load %[[VAL_15]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_17:.*]] = llvm.add %[[VAL_14]], %[[VAL_16]]  : i64
// CHECK:           llvm.return %[[VAL_17]] : i64
// CHECK:    }
func.func @test_3(%nd: memref<?x!sycl_nd_item_3_>) -> i64 {
  %0 = sycl.nd_item.get_group_linear_id(%nd) : (memref<?x!sycl_nd_item_3_>) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1 = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.struct<"class.sycl::_V1::range.1", {{.*}}> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
// CHECK-NEXT:    }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>) -> !sycl_range_1_ {
  %0 = sycl.nd_item.get_global_range(%nd) : (memref<?x!sycl_nd_item_1_>) -> !sycl_range_1_
  return %0 : !sycl_range_1_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1 = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 0, %[[VAL_1]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:    }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>, %i: i32) -> i64 {
  %0 = sycl.nd_item.get_global_range(%nd, %i) : (memref<?x!sycl_nd_item_1_>, i32) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1 = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.struct<"class.sycl::_V1::group.1", {{.*}}> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::group.1", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.struct<"class.sycl::_V1::group.1", {{.*}}>
// CHECK-NEXT:    }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>) -> !sycl_group_1_ {
  %0 = sycl.nd_item.get_group(%nd) : (memref<?x!sycl_nd_item_1_>) -> !sycl_group_1_
  return %0 : !sycl_group_1_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1 = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 3, 0, 0, %[[VAL_1]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:    }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>, %i: i32) -> i64 {
  %0 = sycl.nd_item.get_group(%nd, %i) : (memref<?x!sycl_nd_item_1_>, i32) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1 = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.struct<"class.sycl::_V1::range.1", {{.*}}> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
// CHECK-NEXT:    }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>) -> !sycl_range_1_ {
  %0 = sycl.nd_item.get_group_range(%nd) : (memref<?x!sycl_nd_item_1_>) -> !sycl_range_1_
  return %0 : !sycl_range_1_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1 = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 2, 0, 0, %[[VAL_1]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:    }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>, %i: i32) -> i64 {
  %0 = sycl.nd_item.get_group_range(%nd, %i) : (memref<?x!sycl_nd_item_1_>, i32) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1 = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.struct<"class.sycl::_V1::range.1", {{.*}}> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
// CHECK-NEXT:    }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>) -> !sycl_range_1_ {
  %0 = sycl.nd_item.get_local_range(%nd) : (memref<?x!sycl_nd_item_1_>) -> !sycl_range_1_
  return %0 : !sycl_range_1_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1 = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 0, 0, %[[VAL_1]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:    }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>, %i: i32) -> i64 {
  %0 = sycl.nd_item.get_local_range(%nd, %i) : (memref<?x!sycl_nd_item_1_>, i32) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_item_base_1_1 = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_item_1_1_ = !sycl.item<[1, false], (!sycl_item_base_1_1)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_1_, !sycl_group_1_)>
!sycl_nd_range_1_ = !sycl.nd_range<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.struct<"class.sycl::_V1::nd_range.1", {{.*}}> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::nd_range.1", {{.*}}> : (i32) -> !llvm.ptr
// CHECK-DAG:       %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.1", {{.*}}>
// CHECK-DAG:       %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
// CHECK-DAG:       %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range.1", {{.*}}>
// CHECK:           llvm.store %[[VAL_4]], %[[VAL_5]] : !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>, !llvm.ptr
// CHECK-DAG:       %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.1", {{.*}}>
// CHECK-DAG:       %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
// CHECK-DAG:       %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range.1", {{.*}}>
// CHECK:           llvm.store %[[VAL_7]], %[[VAL_8]] : !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>, !llvm.ptr
// CHECK-DAG:       %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_item.1", {{.*}}>
// CHECK-DAG:       %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK-DAG:       %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range.1", {{.*}}>
// CHECK:           llvm.store %[[VAL_10]], %[[VAL_11]] : !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>, !llvm.ptr
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::nd_range.1", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_12]] : !llvm.struct<"class.sycl::_V1::nd_range.1", {{.*}}>
// CHECK-NEXT:    }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>) -> !sycl_nd_range_1_ {
  %0 = sycl.nd_item.get_nd_range(%nd) : (memref<?x!sycl_nd_item_1_>) -> !sycl_nd_range_1_
  return %0 : !sycl_nd_range_1_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.struct<"class.sycl::_V1::id.1", {{.*}}> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK-NEXT:    }
func.func @test(%group: memref<?x!sycl_group_1_>) -> !sycl_id_1_ {
  %0 = sycl.group.get_group_id(%group) : (memref<?x!sycl_group_1_>) -> !sycl_id_1_
  return %0 : !sycl_id_1_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 3, 0, 0, %[[VAL_1]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:    }
func.func @test(%group: memref<?x!sycl_group_1_>, %i: i32) -> i64 {
  %0 = sycl.group.get_group_id(%group, %i) : (memref<?x!sycl_group_1_>, i32) -> i64
  return %0 : i64
}

// -----

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_group_3_ = !sycl.group<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

module attributes {gpu.container} {
  gpu.module @kernels {
// CHECK-LABEL:     llvm.func @test(
// CHECK-SAME:        %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.struct<"class.sycl::_V1::id.3", {{.*}}> {
// CHECK-NEXT:        %[[VAL_1:.*]] = llvm.mlir.addressof @__spirv_BuiltInLocalInvocationId : !llvm.ptr<1>
// CHECK-NEXT:        %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<1> -> vector<3xi64>
// CHECK-NEXT:        %[[VAL_3:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:        %[[VAL_4:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:        %[[VAL_7:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"class.sycl::_V1::id.3", {{.*}}> : (i64) -> !llvm.ptr
// CHECK-DAG:         %[[VAL_8:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG:         %[[VAL_9:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:         %[[VAL_10:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:             %[[VAL_11:.*]] = llvm.extractelement %[[VAL_2]]{{\[}}%[[VAL_10]] : i32] : vector<3xi64>
// CHECK-NEXT:        %[[VAL_12:.*]] = llvm.getelementptr inbounds %[[VAL_7]][0, 0, 0, %[[VAL_9]]] : (!llvm.ptr, i32) -> !llvm.ptr,  !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
// CHECK-NEXT:        %[[VAL_13:.*]] = llvm.getelementptr %[[VAL_12]]{{\[}}%[[VAL_8]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:        llvm.store %[[VAL_11]], %[[VAL_13]] : i64, !llvm.ptr
// CHECK-DAG:         %[[VAL_14:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG:         %[[VAL_15:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:             %[[VAL_16:.*]] = llvm.extractelement %[[VAL_2]]{{\[}}%[[VAL_15]] : i32] : vector<3xi64>
// CHECK-NEXT:        %[[VAL_17:.*]] = llvm.getelementptr inbounds %[[VAL_7]][0, 0, 0, %[[VAL_14]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
// CHECK-NEXT:        %[[VAL_18:.*]] = llvm.getelementptr %[[VAL_17]]{{\[}}%[[VAL_8]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:        llvm.store %[[VAL_16]], %[[VAL_18]] : i64, !llvm.ptr
// CHECK-DAG:         %[[VAL_19:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-DAG:         %[[VAL_20:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_21:.*]] = llvm.extractelement %[[VAL_2]]{{\[}}%[[VAL_20]] : i32] : vector<3xi64>
// CHECK-NEXT:        %[[VAL_22:.*]] = llvm.getelementptr inbounds %[[VAL_7]][0, 0, 0, %[[VAL_19]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
// CHECK-NEXT:        %[[VAL_23:.*]] = llvm.getelementptr %[[VAL_22]]{{\[}}%[[VAL_8]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:        llvm.store %[[VAL_21]], %[[VAL_23]] : i64, !llvm.ptr
// CHECK-NEXT:        %[[VAL_24:.*]] = llvm.getelementptr %[[VAL_7]]{{\[}}%[[VAL_8]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
// CHECK-NEXT:        %[[VAL_25:.*]] = llvm.load %[[VAL_24]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
// CHECK-NEXT:        llvm.return %[[VAL_25]] : !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
    func.func @test(%group: memref<?x!sycl_group_3_>) -> !sycl_id_3_ {
      %0 = sycl.group.get_local_id(%group) : (memref<?x!sycl_group_3_>) -> !sycl_id_3_
      return %0 : !sycl_id_3_
    }
  }
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>


module attributes {gpu.container} {
  gpu.module @kernels {
// CHECK-LABEL:      llvm.func @test(
// CHECK-SAME:        %[[VAL_0:.*]]: !llvm.ptr
// CHECK-SAME:        %[[VAL_1:.*]]: i32) -> i64 {
// CHECK:             %[[VAL_2:.*]] = llvm.mlir.addressof @__spirv_BuiltInLocalInvocationId : !llvm.ptr<1>
// CHECK:             %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<1> -> vector<3xi64>
// CHECK-DAG:         %[[VAL_4:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:         %[[VAL_5:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:             %[[VAL_8:.*]] = llvm.alloca %[[VAL_4]] x !llvm.struct<"class.sycl::_V1::id.1", {{.*}}> : (i64) -> !llvm.ptr
// CHECK-DAG:         %[[VAL_9:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG:         %[[VAL_10:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:         %[[VAL_11:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_12:.*]] = llvm.extractelement %[[VAL_3]]{{\[}}%[[VAL_11]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_13:.*]] = llvm.getelementptr inbounds %[[VAL_8]][0, 0, 0, %[[VAL_10]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK:             %[[VAL_14:.*]] = llvm.getelementptr %[[VAL_13]]{{\[}}%[[VAL_9]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:             llvm.store %[[VAL_12]], %[[VAL_14]] : i64, !llvm.ptr
// CHECK:             %[[VAL_15:.*]] = llvm.getelementptr %[[VAL_8]]{{\[}}%[[VAL_9]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK:             %[[VAL_16:.*]] = llvm.load %[[VAL_15]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK-DAG:         %[[VAL_17:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:         %[[VAL_18:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:             %[[VAL_21:.*]] = llvm.alloca %[[VAL_17]] x !llvm.struct<"class.sycl::_V1::id.1", {{.*}}> : (i64) -> !llvm.ptr
// CHECK:             llvm.store %[[VAL_16]], %[[VAL_21]] : !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>, !llvm.ptr
// CHECK:             %[[VAL_22:.*]] = llvm.getelementptr inbounds %[[VAL_21]][0, 0, 0, %[[VAL_1]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK:             %[[VAL_23:.*]] = llvm.load %[[VAL_22]] : !llvm.ptr -> i64
// CHECK:             llvm.return %[[VAL_23]] : i64
// CHECK-NEXT:      }
    func.func @test(%group: memref<?x!sycl_group_1_>, %i: i32) -> i64 {
      %0 = sycl.group.get_local_id(%group, %i) : (memref<?x!sycl_group_1_>, i32) -> i64
      return %0 : i64
    }
  }
}

// -----

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_group_3_ = !sycl.group<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.struct<"class.sycl::_V1::range.3", {{.*}}> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::range.3", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.struct<"class.sycl::_V1::range.3", {{.*}}>
// CHECK-NEXT:    }
func.func @test(%group: memref<?x!sycl_group_3_>) -> !sycl_range_3_ {
  %0 = sycl.group.get_local_range(%group) : (memref<?x!sycl_group_3_>) -> !sycl_range_3_
  return %0 : !sycl_range_3_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, %[[VAL_1]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:    }
func.func @test(%group: memref<?x!sycl_group_1_>, %i: i32) -> i64 {
  %0 = sycl.group.get_local_range(%group, %i) : (memref<?x!sycl_group_1_>, i32) -> i64
  return %0 : i64
}

// -----

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_group_3_ = !sycl.group<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.struct<"class.sycl::_V1::range.3", {{.*}}> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::range.3", {{.*}}>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.struct<"class.sycl::_V1::range.3", {{.*}}>
// CHECK-NEXT:    }
func.func @test(%group: memref<?x!sycl_group_3_>) -> !sycl_range_3_ {
  %0 = sycl.group.get_max_local_range(%group) : (memref<?x!sycl_group_3_>) -> !sycl_range_3_
  return %0 : !sycl_range_3_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_group_2_ = !sycl.group<[2], (!sycl_range_2_, !sycl_range_2_, !sycl_range_2_, !sycl_id_2_)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_group_3_ = !sycl.group<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

// CHECK-LABEL:   llvm.func @test_1(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 3, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i64
// CHECK-NEXT:      llvm.return %[[VAL_2]] : i64
// CHECK-NEXT:    }
func.func @test_1(%group: memref<?x!sycl_group_1_>) -> i64 {
  %0 = sycl.group.get_group_linear_id(%group) : (memref<?x!sycl_group_1_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 3, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 3, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.add %[[VAL_5]], %[[VAL_7]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_8]] : i64
// CHECK-NEXT:    }
func.func @test_2(%group: memref<?x!sycl_group_2_>) -> i64 {
  %0 = sycl.group.get_group_linear_id(%group) : (memref<?x!sycl_group_2_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 3, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.mul %[[VAL_5]], %[[VAL_7]]  : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 3, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.mul %[[VAL_10]], %[[VAL_7]]  : i64
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.add %[[VAL_8]], %[[VAL_11]]  : i64
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 3, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.add %[[VAL_12]], %[[VAL_14]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_15]] : i64
// CHECK-NEXT:    }
func.func @test_3(%group: memref<?x!sycl_group_3_>) -> i64 {
  %0 = sycl.group.get_group_linear_id(%group) : (memref<?x!sycl_group_3_>) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_group_2_ = !sycl.group<[2], (!sycl_range_2_, !sycl_range_2_, !sycl_range_2_, !sycl_id_2_)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_group_3_ = !sycl.group<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

module attributes {gpu.container} {
  gpu.module @kernels {
// CHECK-LABEL:   llvm.func @test_1(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK:             %[[VAL_1:.*]] = llvm.mlir.addressof @__spirv_BuiltInLocalInvocationId : !llvm.ptr<1>
// CHECK:             %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<1> -> vector<3xi64>
// CHECK-DAG:         %[[VAL_3:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:         %[[VAL_4:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:             %[[VAL_7:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"class.sycl::_V1::id.1", {{.*}}> : (i64) -> !llvm.ptr
// CHECK-DAG:         %[[VAL_8:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG:         %[[VAL_9:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:         %[[VAL_10:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_11:.*]] = llvm.extractelement %[[VAL_2]]{{\[}}%[[VAL_10]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_12:.*]] = llvm.getelementptr inbounds %[[VAL_7]][0, 0, 0, %[[VAL_9]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK:             %[[VAL_13:.*]] = llvm.getelementptr %[[VAL_12]]{{\[}}%[[VAL_8]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:             llvm.store %[[VAL_11]], %[[VAL_13]] : i64, !llvm.ptr
// CHECK:             %[[VAL_14:.*]] = llvm.getelementptr %[[VAL_7]]{{\[}}%[[VAL_8]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK:             %[[VAL_15:.*]] = llvm.load %[[VAL_14]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK-DAG:         %[[VAL_16:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:         %[[VAL_17:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:             %[[VAL_20:.*]] = llvm.alloca %[[VAL_16]] x !llvm.struct<"class.sycl::_V1::id.1", {{.*}}> : (i64) -> !llvm.ptr
// CHECK:             llvm.store %[[VAL_15]], %[[VAL_20]] : !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>, !llvm.ptr
// CHECK:             %[[VAL_21:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_22:.*]] = llvm.getelementptr inbounds %[[VAL_20]][0, 0, 0, %[[VAL_21]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
// CHECK:             %[[VAL_23:.*]] = llvm.load %[[VAL_22]] : !llvm.ptr -> i64
// CHECK:             llvm.return %[[VAL_23]] : i64
// CHECK-NEXT:     }
    func.func @test_1(%group: memref<?x!sycl_group_1_>) -> i64 {
      %0 = sycl.group.get_local_linear_id(%group) : (memref<?x!sycl_group_1_>) -> i64
      return %0 : i64
    }

// CHECK-NEXT:      llvm.func @test_2(%[[VAL_24:.*]]: !llvm.ptr) -> i64 {
// CHECK:             %[[VAL_25:.*]] = llvm.mlir.addressof @__spirv_BuiltInLocalInvocationId : !llvm.ptr<1>
// CHECK:             %[[VAL_26:.*]] = llvm.load %[[VAL_25]] : !llvm.ptr<1> -> vector<3xi64>
// CHECK-DAG:         %[[VAL_27:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:         %[[VAL_28:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:             %[[VAL_31:.*]] = llvm.alloca %[[VAL_27]] x !llvm.struct<"class.sycl::_V1::id.2", {{.*}}> : (i64) -> !llvm.ptr
// CHECK-DAG:         %[[VAL_32:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG:         %[[VAL_33:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:         %[[VAL_34:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:             %[[VAL_35:.*]] = llvm.extractelement %[[VAL_26]]{{\[}}%[[VAL_34]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_36:.*]] = llvm.getelementptr inbounds %[[VAL_31]][0, 0, 0, %[[VAL_33]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.2", {{.*}}>
// CHECK:             %[[VAL_37:.*]] = llvm.getelementptr %[[VAL_36]]{{\[}}%[[VAL_32]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:             llvm.store %[[VAL_35]], %[[VAL_37]] : i64, !llvm.ptr
// CHECK-DAG:         %[[VAL_38:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG:         %[[VAL_39:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_40:.*]] = llvm.extractelement %[[VAL_26]]{{\[}}%[[VAL_39]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_41:.*]] = llvm.getelementptr inbounds %[[VAL_31]][0, 0, 0, %[[VAL_38]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.2", {{.*}}>
// CHECK:             %[[VAL_42:.*]] = llvm.getelementptr %[[VAL_41]]{{\[}}%[[VAL_32]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:             llvm.store %[[VAL_40]], %[[VAL_42]] : i64, !llvm.ptr
// CHECK:             %[[VAL_43:.*]] = llvm.getelementptr %[[VAL_31]]{{\[}}%[[VAL_32]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.2", {{.*}}>
// CHECK:             %[[VAL_44:.*]] = llvm.load %[[VAL_43]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::id.2", {{.*}}>
// CHECK-DAG:         %[[VAL_45:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:         %[[VAL_46:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:             %[[VAL_49:.*]] = llvm.alloca %[[VAL_45]] x !llvm.struct<"class.sycl::_V1::id.2", {{.*}}> : (i64) -> !llvm.ptr
// CHECK:             llvm.store %[[VAL_44]], %[[VAL_49]] : !llvm.struct<"class.sycl::_V1::id.2", {{.*}}>, !llvm.ptr
// CHECK:             %[[VAL_50:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_51:.*]] = llvm.getelementptr inbounds %[[VAL_49]][0, 0, 0, %[[VAL_50]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.2", {{.*}}>
// CHECK:             %[[VAL_52:.*]] = llvm.load %[[VAL_51]] : !llvm.ptr -> i64
// CHECK:             %[[VAL_53:.*]] = llvm.getelementptr inbounds %[[VAL_24]][0, 1, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.2", {{.*}}>
// CHECK:             %[[VAL_54:.*]] = llvm.load %[[VAL_53]] : !llvm.ptr -> i64
// CHECK:             %[[VAL_55:.*]] = llvm.mul %[[VAL_52]], %[[VAL_54]]  : i64
// CHECK:             %[[VAL_56:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:             %[[VAL_57:.*]] = llvm.getelementptr inbounds %[[VAL_49]][0, 0, 0, %[[VAL_56]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.2", {{.*}}>
// CHECK:             %[[VAL_58:.*]] = llvm.load %[[VAL_57]] : !llvm.ptr -> i64
// CHECK:             %[[VAL_59:.*]] = llvm.add %[[VAL_55]], %[[VAL_58]]  : i64
// CHECK:             llvm.return %[[VAL_59]] : i64
// CHECK-NEXT:      }
    func.func @test_2(%group: memref<?x!sycl_group_2_>) -> i64 {
      %0 = sycl.group.get_local_linear_id(%group) : (memref<?x!sycl_group_2_>) -> i64
      return %0 : i64
    }

// CHECK-NEXT:      llvm.func @test_3(%[[VAL_60:.*]]: !llvm.ptr) -> i64 {
// CHECK:             %[[VAL_61:.*]] = llvm.mlir.addressof @__spirv_BuiltInLocalInvocationId : !llvm.ptr<1>
// CHECK:             %[[VAL_62:.*]] = llvm.load %[[VAL_61]] : !llvm.ptr<1> -> vector<3xi64>
// CHECK-DAG:         %[[VAL_63:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:         %[[VAL_64:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:             %[[VAL_67:.*]] = llvm.alloca %[[VAL_63]] x !llvm.struct<"class.sycl::_V1::id.3", {{.*}}> : (i64) -> !llvm.ptr
// CHECK-DAG:         %[[VAL_68:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG:         %[[VAL_69:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:         %[[VAL_70:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:             %[[VAL_71:.*]] = llvm.extractelement %[[VAL_62]]{{\[}}%[[VAL_70]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_72:.*]] = llvm.getelementptr inbounds %[[VAL_67]][0, 0, 0, %[[VAL_69]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
// CHECK:             %[[VAL_73:.*]] = llvm.getelementptr %[[VAL_72]]{{\[}}%[[VAL_68]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:             llvm.store %[[VAL_71]], %[[VAL_73]] : i64, !llvm.ptr
// CHECK-DAG:         %[[VAL_74:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG:         %[[VAL_75:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:             %[[VAL_76:.*]] = llvm.extractelement %[[VAL_62]]{{\[}}%[[VAL_75]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_77:.*]] = llvm.getelementptr inbounds %[[VAL_67]][0, 0, 0, %[[VAL_74]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
// CHECK:             %[[VAL_78:.*]] = llvm.getelementptr %[[VAL_77]]{{\[}}%[[VAL_68]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:             llvm.store %[[VAL_76]], %[[VAL_78]] : i64, !llvm.ptr
// CHECK-DAG:         %[[VAL_79:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-DAG:         %[[VAL_80:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_81:.*]] = llvm.extractelement %[[VAL_62]]{{\[}}%[[VAL_80]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_82:.*]] = llvm.getelementptr inbounds %[[VAL_67]][0, 0, 0, %[[VAL_79]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
// CHECK:             %[[VAL_83:.*]] = llvm.getelementptr %[[VAL_82]]{{\[}}%[[VAL_68]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK:             llvm.store %[[VAL_81]], %[[VAL_83]] : i64, !llvm.ptr
// CHECK:             %[[VAL_84:.*]] = llvm.getelementptr %[[VAL_67]]{{\[}}%[[VAL_68]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
// CHECK:             %[[VAL_85:.*]] = llvm.load %[[VAL_84]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
// CHECK-DAG:         %[[VAL_86:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:         %[[VAL_87:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:             %[[VAL_90:.*]] = llvm.alloca %[[VAL_86]] x !llvm.struct<"class.sycl::_V1::id.3", {{.*}}> : (i64) -> !llvm.ptr
// CHECK:             llvm.store %[[VAL_85]], %[[VAL_90]] : !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>, !llvm.ptr
// CHECK:             %[[VAL_91:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_92:.*]] = llvm.getelementptr inbounds %[[VAL_90]][0, 0, 0, %[[VAL_91]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
// CHECK:             %[[VAL_93:.*]] = llvm.load %[[VAL_92]] : !llvm.ptr -> i64
// CHECK:             %[[VAL_94:.*]] = llvm.getelementptr inbounds %[[VAL_60]][0, 1, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK:             %[[VAL_95:.*]] = llvm.load %[[VAL_94]] : !llvm.ptr -> i64
// CHECK:             %[[VAL_96:.*]] = llvm.mul %[[VAL_93]], %[[VAL_95]]  : i64
// CHECK:             %[[VAL_97:.*]] = llvm.getelementptr inbounds %[[VAL_60]][0, 1, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK:             %[[VAL_98:.*]] = llvm.load %[[VAL_97]] : !llvm.ptr -> i64
// CHECK:             %[[VAL_99:.*]] = llvm.mul %[[VAL_96]], %[[VAL_98]]  : i64
// CHECK:             %[[VAL_100:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:             %[[VAL_101:.*]] = llvm.getelementptr inbounds %[[VAL_90]][0, 0, 0, %[[VAL_100]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
// CHECK:             %[[VAL_102:.*]] = llvm.load %[[VAL_101]] : !llvm.ptr -> i64
// CHECK:             %[[VAL_103:.*]] = llvm.mul %[[VAL_102]], %[[VAL_98]]  : i64
// CHECK:             %[[VAL_104:.*]] = llvm.add %[[VAL_99]], %[[VAL_103]]  : i64
// CHECK:             %[[VAL_105:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:             %[[VAL_106:.*]] = llvm.getelementptr inbounds %[[VAL_90]][0, 0, 0, %[[VAL_105]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
// CHECK:             %[[VAL_107:.*]] = llvm.load %[[VAL_106]] : !llvm.ptr -> i64
// CHECK:             %[[VAL_108:.*]] = llvm.add %[[VAL_104]], %[[VAL_107]]  : i64
// CHECK:             llvm.return %[[VAL_108]] : i64
// CHECK-NEXT:      }
    func.func @test_3(%group: memref<?x!sycl_group_3_>) -> i64 {
      %0 = sycl.group.get_local_linear_id(%group) : (memref<?x!sycl_group_3_>) -> i64
      return %0 : i64
    }
  }
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_group_2_ = !sycl.group<[2], (!sycl_range_2_, !sycl_range_2_, !sycl_range_2_, !sycl_id_2_)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_group_3_ = !sycl.group<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

// CHECK-LABEL:   llvm.func @test_1(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mul %[[VAL_3]], %[[VAL_1]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_4]] : i64
// CHECK-NEXT:    }
func.func @test_1(%group: memref<?x!sycl_group_1_>) -> i64 {
  %0 = sycl.group.get_group_linear_range(%group) : (memref<?x!sycl_group_1_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mul %[[VAL_3]], %[[VAL_1]]  : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_4]], %[[VAL_6]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_7]] : i64
// CHECK-NEXT:    }
func.func @test_2(%group: memref<?x!sycl_group_2_>) -> i64 {
  %0 = sycl.group.get_group_linear_range(%group) : (memref<?x!sycl_group_2_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mul %[[VAL_3]], %[[VAL_1]]  : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_4]], %[[VAL_6]]  : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.mul %[[VAL_7]], %[[VAL_9]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_10]] : i64
// CHECK-NEXT:    }
func.func @test_3(%group: memref<?x!sycl_group_3_>) -> i64 {
  %0 = sycl.group.get_group_linear_range(%group) : (memref<?x!sycl_group_3_>) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_group_2_ = !sycl.group<[2], (!sycl_range_2_, !sycl_range_2_, !sycl_range_2_, !sycl_id_2_)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_group_3_ = !sycl.group<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

// CHECK-LABEL:   llvm.func @test_1(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.1", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mul %[[VAL_3]], %[[VAL_1]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_4]] : i64
// CHECK-NEXT:    }
func.func @test_1(%group: memref<?x!sycl_group_1_>) -> i64 {
  %0 = sycl.group.get_local_linear_range(%group) : (memref<?x!sycl_group_1_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mul %[[VAL_3]], %[[VAL_1]]  : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.2", {{.*}}>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_4]], %[[VAL_6]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_7]] : i64
// CHECK-NEXT:    }
func.func @test_2(%group: memref<?x!sycl_group_2_>) -> i64 {
  %0 = sycl.group.get_local_linear_range(%group) : (memref<?x!sycl_group_2_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mul %[[VAL_3]], %[[VAL_1]]  : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_4]], %[[VAL_6]]  : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::group.3", {{.*}}>
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.mul %[[VAL_7]], %[[VAL_9]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_10]] : i64
// CHECK-NEXT:    }
func.func @test_3(%group: memref<?x!sycl_group_3_>) -> i64 {
  %0 = sycl.group.get_local_linear_range(%group) : (memref<?x!sycl_group_3_>) -> i64
  return %0 : i64
}
