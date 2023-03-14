// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm %s | FileCheck %s

//===-------------------------------------------------------------------------------------------------===//
// sycl.range.get with reference result type
//===-------------------------------------------------------------------------------------------------===//

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[RANGE3:.*]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> !llvm.ptr<i64, 4> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[RANGE3]]>, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.addrspacecast %[[VAL_2]] : !llvm.ptr<i64> to !llvm.ptr<i64, 4>
// CHECK-NEXT:      llvm.return %[[VAL_3]] : !llvm.ptr<i64, 4>
// CHECK-NEXT:    }
func.func @test(%range: memref<?x!sycl_range_3_>, %idx: i32) -> memref<?xi64, 4> {
  %0 = sycl.range.get %range[%idx] { ArgumentTypes = [memref<?x!sycl_range_3_>, i32], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"range" }  : (memref<?x!sycl_range_3_>, i32) -> memref<?xi64, 4>
  return %0 : memref<?xi64, 4>
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.id.get with reference result type
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ID3:.*]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> !llvm.ptr<i64, 4> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[ID3]]>, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.addrspacecast %[[VAL_2]] : !llvm.ptr<i64> to !llvm.ptr<i64, 4>
// CHECK-NEXT:      llvm.return %[[VAL_3]] : !llvm.ptr<i64, 4>
// CHECK-NEXT:    }
func.func @test(%id: memref<?x!sycl_id_3_>, %idx: i32) -> memref<?xi64, 4> {
  %0 = sycl.id.get %id[%idx] { ArgumentTypes = [memref<?x!sycl_id_3_>, i32], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"id" }  : (memref<?x!sycl_id_3_>, i32) -> memref<?xi64, 4>
  return %0 : memref<?xi64, 4>
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.accessor.subscript with scalar offset and 1D accessor//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR1:.*]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i64) -> !llvm.ptr<i32, 4> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0] : (!llvm.ptr<[[ACCESSOR1]]>) -> !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_3]][%[[VAL_1]]] : (!llvm.ptr<i32, 1>, i64) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.addrspacecast %[[VAL_4]] : !llvm.ptr<i32, 1> to !llvm.ptr<i32, 4>
// CHECK-NEXT:      llvm.return %[[VAL_5]] : !llvm.ptr<i32, 4>
// CHECK-NEXT:    }
func.func @test(%acc: memref<?x!sycl_accessor_1_i32_rw_gb>, %idx: i64) -> memref<?xi32, 4> {
  %0 = sycl.accessor.subscript %acc[%idx] { ArgumentTypes = [memref<?x!sycl_accessor_1_i32_rw_gb>, i64], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"id" }  : (memref<?x!sycl_accessor_1_i32_rw_gb>, i64) -> memref<?xi32, 4>
  return %0 : memref<?xi32, 4>
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.accessor.subscript with id offset
//===-------------------------------------------------------------------------------------------------===//

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

// CHECK-LABEL:   llvm.func @test_1(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR1]]>,
// CHECK-SAME:                      %[[VAL_1:.*]]: !llvm.ptr<[[ID1:.*]]>) -> !llvm.ptr<i32, 4> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ACCESSOR1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr<[[ID1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.add %[[VAL_7]], %[[VAL_6]] : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0] : (!llvm.ptr<[[ACCESSOR1]]>) -> !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_10]][%[[VAL_8]]] : (!llvm.ptr<i32, 1>, i64) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.addrspacecast %[[VAL_11]] : !llvm.ptr<i32, 1> to !llvm.ptr<i32, 4>
// CHECK-NEXT:      llvm.return %[[VAL_12]] : !llvm.ptr<i32, 4>
// CHECK-NEXT:    }
func.func @test_1(%acc: memref<?x!sycl_accessor_1_i32_rw_gb>, %idx: memref<?x!sycl_id_1_>) -> memref<?xi32, 4> {
  %0 = sycl.accessor.subscript %acc[%idx] { ArgumentTypes = [memref<?x!sycl_accessor_1_i32_rw_gb>, memref<?x!sycl_id_1_>], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"id" }  : (memref<?x!sycl_accessor_1_i32_rw_gb>, memref<?x!sycl_id_1_>) -> memref<?xi32, 4>
  return %0 : memref<?xi32, 4>
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR2:.*]]>,
// CHECK-SAME:                      %[[VAL_1:.*]]: !llvm.ptr<[[ID2:.*]]>) -> !llvm.ptr<i32, 4> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ACCESSOR2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr<[[ID2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.add %[[VAL_7]], %[[VAL_6]] : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 1] : (!llvm.ptr<[[ACCESSOR2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 1] : (!llvm.ptr<[[ID2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.mul %[[VAL_8]], %[[VAL_10]] : i64
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.add %[[VAL_13]], %[[VAL_12]] : i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0] : (!llvm.ptr<[[ACCESSOR2]]>) -> !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_16:.*]] = llvm.load %[[VAL_15]] : !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_17:.*]] = llvm.getelementptr inbounds %[[VAL_16]][%[[VAL_14]]] : (!llvm.ptr<i32, 1>, i64) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:      %[[VAL_18:.*]] = llvm.addrspacecast %[[VAL_17]] : !llvm.ptr<i32, 1> to !llvm.ptr<i32, 4>
// CHECK-NEXT:      llvm.return %[[VAL_18]] : !llvm.ptr<i32, 4>
// CHECK-NEXT:    }
func.func @test_2(%acc: memref<?x!sycl_accessor_2_i32_rw_gb>, %idx: memref<?x!sycl_id_2_>) -> memref<?xi32, 4> {
  %0 = sycl.accessor.subscript %acc[%idx] { ArgumentTypes = [memref<?x!sycl_accessor_2_i32_rw_gb>, memref<?x!sycl_id_2_>], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"id" }  : (memref<?x!sycl_accessor_2_i32_rw_gb>, memref<?x!sycl_id_2_>) -> memref<?xi32, 4>
  return %0 : memref<?xi32, 4>
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR3:.*]]>,
// CHECK-SAME:                      %[[VAL_1:.*]]: !llvm.ptr<[[ID3]]>) -> !llvm.ptr<i32, 4> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ACCESSOR3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr<[[ID3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.add %[[VAL_7]], %[[VAL_6]] : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 1] : (!llvm.ptr<[[ACCESSOR3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 1] : (!llvm.ptr<[[ID3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.mul %[[VAL_8]], %[[VAL_10]] : i64
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.add %[[VAL_13]], %[[VAL_12]] : i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 2] : (!llvm.ptr<[[ACCESSOR3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_16:.*]] = llvm.load %[[VAL_15]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_17:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 2] : (!llvm.ptr<[[ID3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_18:.*]] = llvm.load %[[VAL_17]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_19:.*]] = llvm.mul %[[VAL_14]], %[[VAL_16]] : i64
// CHECK-NEXT:      %[[VAL_20:.*]] = llvm.add %[[VAL_19]], %[[VAL_18]] : i64
// CHECK-NEXT:      %[[VAL_21:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0] : (!llvm.ptr<[[ACCESSOR3]]>) -> !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_22:.*]] = llvm.load %[[VAL_21]] : !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_23:.*]] = llvm.getelementptr inbounds %[[VAL_22]][%[[VAL_20]]] : (!llvm.ptr<i32, 1>, i64) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:      %[[VAL_24:.*]] = llvm.addrspacecast %[[VAL_23]] : !llvm.ptr<i32, 1> to !llvm.ptr<i32, 4>
// CHECK-NEXT:      llvm.return %[[VAL_24]] : !llvm.ptr<i32, 4>
// CHECK-NEXT:    }
func.func @test_3(%acc: memref<?x!sycl_accessor_3_i32_rw_gb>, %idx: memref<?x!sycl_id_3_>) -> memref<?xi32, 4> {
  %0 = sycl.accessor.subscript %acc[%idx] { ArgumentTypes = [memref<?x!sycl_accessor_3_i32_rw_gb>, memref<?x!sycl_id_3_>], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"id" }  : (memref<?x!sycl_accessor_3_i32_rw_gb>, memref<?x!sycl_id_3_>) -> memref<?xi32, 4>
  return %0 : memref<?xi32, 4>
}
