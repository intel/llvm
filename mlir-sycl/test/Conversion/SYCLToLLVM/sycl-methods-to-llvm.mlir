// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm="use-bare-ptr-call-conv" %s | FileCheck %s

//===-------------------------------------------------------------------------------------------------===//
// sycl.range.get with scalar result type
//===-------------------------------------------------------------------------------------------------===//

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[RANGE3:.*]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[RANGE3]]>, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:      llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:    }
func.func @test(%range: memref<?x!sycl_range_3_>, %idx: i32) -> i64 {
  %0 = sycl.range.get %range[%idx] { ArgumentTypes = [memref<?x!sycl_range_3_>, i32], FunctionName = @"get", MangledFunctionName = @"get", TypeName = @"range" }  : (memref<?x!sycl_range_3_>, i32) -> i64
  return %0 : i64
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.range.get with reference result type
//===-------------------------------------------------------------------------------------------------===//

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[RANGE3]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> !llvm.ptr<i64> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[RANGE3]]>, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:    }
func.func @test(%range: memref<?x!sycl_range_3_>, %idx: i32) -> memref<?xi64> {
  %0 = sycl.range.get %range[%idx] { ArgumentTypes = [memref<?x!sycl_range_3_>, i32], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"range" }  : (memref<?x!sycl_range_3_>, i32) -> memref<?xi64>
  return %0 : memref<?xi64>
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.range.size
//===-------------------------------------------------------------------------------------------------===//

!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

// CHECK-LABEL:   llvm.func @test_1(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[RANGE1:.*]]>) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0] : (!llvm.ptr<[[RANGE1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.muli %[[VAL_1]], %[[VAL_3]] : i64
// CHECK-NEXT:      llvm.return %[[VAL_4]] : i64
// CHECK-NEXT:    }
func.func @test_1(%range: memref<?x!sycl_range_1_>) -> i64 {
  %0 = sycl.range.size(%range) { ArgumentTypes = [memref<?x!sycl_range_1_>], FunctionName = @"size", MangledFunctionName = @"size", TypeName = @"range" }  : (memref<?x!sycl_range_1_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[RANGE2:.*]]>) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0] : (!llvm.ptr<[[RANGE2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.muli %[[VAL_1]], %[[VAL_3]] : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1] : (!llvm.ptr<[[RANGE2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_7:.*]] = arith.muli %[[VAL_4]], %[[VAL_6]] : i64
// CHECK-NEXT:      llvm.return %[[VAL_7]] : i64
// CHECK-NEXT:    }
func.func @test_2(%range: memref<?x!sycl_range_2_>) -> i64 {
  %0 = sycl.range.size(%range) { ArgumentTypes = [memref<?x!sycl_range_2_>], FunctionName = @"size", MangledFunctionName = @"size", TypeName = @"range" }  : (memref<?x!sycl_range_2_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[RANGE3]]>) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0] : (!llvm.ptr<[[RANGE3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.muli %[[VAL_1]], %[[VAL_3]] : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1] : (!llvm.ptr<[[RANGE3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_7:.*]] = arith.muli %[[VAL_4]], %[[VAL_6]] : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 2] : (!llvm.ptr<[[RANGE3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_10:.*]] = arith.muli %[[VAL_7]], %[[VAL_9]] : i64
// CHECK-NEXT:      llvm.return %[[VAL_10]] : i64
// CHECK-NEXT:    }
func.func @test_3(%range: memref<?x!sycl_range_3_>) -> i64 {
  %0 = sycl.range.size(%range) { ArgumentTypes = [memref<?x!sycl_range_3_>], FunctionName = @"size", MangledFunctionName = @"size", TypeName = @"range" }  : (memref<?x!sycl_range_3_>) -> i64
  return %0 : i64
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.id.get with scalar result type
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ID3:.*]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[ID3]]>, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:      llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:    }
func.func @test(%id: memref<?x!sycl_id_3_>, %idx: i32) -> i64 {
  %0 = sycl.id.get %id[%idx] { ArgumentTypes = [memref<?x!sycl_id_3_>, i32], FunctionName = @"get", MangledFunctionName = @"get", TypeName = @"id" }  : (memref<?x!sycl_id_3_>, i32) -> i64
  return %0 : i64
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.id.get with reference result type
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ID3]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> !llvm.ptr<i64> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[ID3]]>, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:    }
func.func @test(%id: memref<?x!sycl_id_3_>, %idx: i32) -> memref<?xi64> {
  %0 = sycl.id.get %id[%idx] { ArgumentTypes = [memref<?x!sycl_id_3_>, i32], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"id" }  : (memref<?x!sycl_id_3_>, i32) -> memref<?xi64>
  return %0 : memref<?xi64>
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.accessor.subscript with scalar offset and 1D accessor
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR1:.*]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i64) -> !llvm.ptr<i32, 1> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0] : (!llvm.ptr<[[ACCESSOR1]]>) -> !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_3]][%[[VAL_1]]] : (!llvm.ptr<i32, 1>, i64) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:      llvm.return %[[VAL_4]] : !llvm.ptr<i32, 1>
// CHECK-NEXT:    }
func.func @test(%acc: memref<?x!sycl_accessor_1_i32_rw_gb>, %idx: i64) -> memref<?xi32, 1> {
  %0 = sycl.accessor.subscript %acc[%idx] { ArgumentTypes = [memref<?x!sycl_accessor_1_i32_rw_gb>, i64], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"id" }  : (memref<?x!sycl_accessor_1_i32_rw_gb>, i64) -> memref<?xi32, 1>
  return %0 : memref<?xi32, 1>
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.accessor.subscript with scalar offset and ND accessor
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_accessor_impl_device_2_ = !sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>
!sycl_accessor_2_i32_rw_gb = !sycl.accessor<[2, i32, read_write, global_buffer], (!sycl_accessor_impl_device_2_, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_accessor_impl_device_3_ = !sycl.accessor_impl_device<[3], (!sycl_id_3_, !sycl_range_3_, !sycl_range_3_)>
!sycl_accessor_3_i32_rw_gb = !sycl.accessor<[3, i32, read_write, global_buffer], (!sycl_accessor_impl_device_3_, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_accessor_subscript_2_ = !sycl.accessor_subscript<[2], (!sycl_id_2_, !sycl_accessor_2_i32_rw_gb)>
!sycl_accessor_subscript_3_ = !sycl.accessor_subscript<[3], (!sycl_id_3_, !sycl_accessor_3_i32_rw_gb)>

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR2:.*]]>, %[[VAL_1:.*]]: i64) -> !llvm.[[ACCESSORSUBS2:.*]] {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.undef : !llvm.[[ACCESSORSUBS2]]
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_2]][0, 0, 0, 0] : !llvm.[[ACCESSORSUBS2]]
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.constant 0 : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.load %[[VAL_0]] : !llvm.ptr<[[ACCESSOR2]]>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_3]][1] : !llvm.[[ACCESSORSUBS2]]
// CHECK-NEXT:      llvm.return %[[VAL_6]] : !llvm.[[ACCESSORSUBS2]]
// CHECK-NEXT:    }
func.func @test_2(%acc: memref<?x!sycl_accessor_2_i32_rw_gb>, %idx: i64) -> !sycl_accessor_subscript_2_ {
  %0 = sycl.accessor.subscript %acc[%idx] { ArgumentTypes = [memref<?x!sycl_accessor_2_i32_rw_gb>, i64], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"id" }  : (memref<?x!sycl_accessor_2_i32_rw_gb>, i64) -> !sycl_accessor_subscript_2_
  return %0 : !sycl_accessor_subscript_2_
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR3:.*]]>, %[[VAL_1:.*]]: i64) -> !llvm.[[ACCESSORSUBS3:.*]] {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.undef : !llvm.[[ACCESSORSUBS3]]
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_2]][0, 0, 0, 0] : !llvm.[[ACCESSORSUBS3]]
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.constant 0 : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_3]][0, 0, 0, 1] : !llvm.[[ACCESSORSUBS3]]
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_0]] : !llvm.ptr<[[ACCESSOR3]]>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_5]][1] : !llvm.[[ACCESSORSUBS3]]
// CHECK-NEXT:      llvm.return %[[VAL_7]] : !llvm.[[ACCESSORSUBS3]]
// CHECK-NEXT:    }
func.func @test_3(%acc: memref<?x!sycl_accessor_3_i32_rw_gb>, %idx: i64) -> !sycl_accessor_subscript_3_ {
  %0 = sycl.accessor.subscript %acc[%idx] { ArgumentTypes = [memref<?x!sycl_accessor_3_i32_rw_gb>, i64], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"id" }  : (memref<?x!sycl_accessor_3_i32_rw_gb>, i64) -> !sycl_accessor_subscript_3_
  return %0 : !sycl_accessor_subscript_3_
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.accessor.subscript with id offset
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_accessor_impl_device_2_ = !sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>
!sycl_accessor_2_i32_rw_gb = !sycl.accessor<[2, i32, read_write, global_buffer], (!sycl_accessor_impl_device_2_, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_accessor_impl_device_3_ = !sycl.accessor_impl_device<[3], (!sycl_id_3_, !sycl_range_3_, !sycl_range_3_)>
!sycl_accessor_3_i32_rw_gb = !sycl.accessor<[3, i32, read_write, global_buffer], (!sycl_accessor_impl_device_3_, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK-LABEL:   llvm.func @test_1(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR1]]>,
// CHECK-SAME:                      %[[VAL_1:.*]]: !llvm.ptr<[[ID1:.*]]>) -> !llvm.ptr<i32, 1> {
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ACCESSOR1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr<[[ID1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_7:.*]] = arith.muli %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %[[VAL_6]] : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0] : (!llvm.ptr<[[ACCESSOR1]]>) -> !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_10]][%[[VAL_8]]] : (!llvm.ptr<i32, 1>, i64) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:      llvm.return %[[VAL_11]] : !llvm.ptr<i32, 1>
// CHECK-NEXT:    }
func.func @test_1(%acc: memref<?x!sycl_accessor_1_i32_rw_gb>, %idx: memref<?x!sycl_id_1_>) -> memref<?xi32, 1> {
  %0 = sycl.accessor.subscript %acc[%idx] { ArgumentTypes = [memref<?x!sycl_accessor_1_i32_rw_gb>, memref<?x!sycl_id_1_>], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"id" }  : (memref<?x!sycl_accessor_1_i32_rw_gb>, memref<?x!sycl_id_1_>) -> memref<?xi32, 1>
  return %0 : memref<?xi32, 1>
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR2]]>,
// CHECK-SAME:                      %[[VAL_1:.*]]: !llvm.ptr<[[ID2:.*]]>) -> !llvm.ptr<i32, 1> {
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ACCESSOR2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr<[[ID2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_7:.*]] = arith.muli %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %[[VAL_6]] : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 1] : (!llvm.ptr<[[ACCESSOR2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 1] : (!llvm.ptr<[[ID2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_13:.*]] = arith.muli %[[VAL_8]], %[[VAL_10]] : i64
// CHECK-NEXT:      %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_12]] : i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0] : (!llvm.ptr<[[ACCESSOR2]]>) -> !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_16:.*]] = llvm.load %[[VAL_15]] : !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_17:.*]] = llvm.getelementptr inbounds %[[VAL_16]][%[[VAL_14]]] : (!llvm.ptr<i32, 1>, i64) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:      llvm.return %[[VAL_17]] : !llvm.ptr<i32, 1>
// CHECK-NEXT:    }
func.func @test_2(%acc: memref<?x!sycl_accessor_2_i32_rw_gb>, %idx: memref<?x!sycl_id_2_>) -> memref<?xi32, 1> {
  %0 = sycl.accessor.subscript %acc[%idx] { ArgumentTypes = [memref<?x!sycl_accessor_2_i32_rw_gb>, memref<?x!sycl_id_2_>], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"id" }  : (memref<?x!sycl_accessor_2_i32_rw_gb>, memref<?x!sycl_id_2_>) -> memref<?xi32, 1>
  return %0 : memref<?xi32, 1>
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR3:.*]]>,
// CHECK-SAME:                      %[[VAL_1:.*]]: !llvm.ptr<[[ID3]]>) -> !llvm.ptr<i32, 1> {
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ACCESSOR3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr<[[ID3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_7:.*]] = arith.muli %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %[[VAL_6]] : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 1] : (!llvm.ptr<[[ACCESSOR3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 1] : (!llvm.ptr<[[ID3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_13:.*]] = arith.muli %[[VAL_8]], %[[VAL_10]] : i64
// CHECK-NEXT:      %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_12]] : i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 2] : (!llvm.ptr<[[ACCESSOR3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_16:.*]] = llvm.load %[[VAL_15]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_17:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 2] : (!llvm.ptr<[[ID3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_18:.*]] = llvm.load %[[VAL_17]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_19:.*]] = arith.muli %[[VAL_14]], %[[VAL_16]] : i64
// CHECK-NEXT:      %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_18]] : i64
// CHECK-NEXT:      %[[VAL_21:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0] : (!llvm.ptr<[[ACCESSOR3]]>) -> !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_22:.*]] = llvm.load %[[VAL_21]] : !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_23:.*]] = llvm.getelementptr inbounds %[[VAL_22]][%[[VAL_20]]] : (!llvm.ptr<i32, 1>, i64) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:      llvm.return %[[VAL_23]] : !llvm.ptr<i32, 1>
// CHECK-NEXT:    }
func.func @test_3(%acc: memref<?x!sycl_accessor_3_i32_rw_gb>, %idx: memref<?x!sycl_id_3_>) -> memref<?xi32, 1> {
  %0 = sycl.accessor.subscript %acc[%idx] { ArgumentTypes = [memref<?x!sycl_accessor_3_i32_rw_gb>, memref<?x!sycl_id_3_>], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"id" }  : (memref<?x!sycl_accessor_3_i32_rw_gb>, memref<?x!sycl_id_3_>) -> memref<?xi32, 1>
  return %0 : memref<?xi32, 1>
}
