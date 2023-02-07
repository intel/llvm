// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm="use-bare-ptr-call-conv" %s | FileCheck %s

//===-------------------------------------------------------------------------------------------------===//
// sycl.nd_range.get_global_range
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_nd_range_3_ = !sycl.nd_range<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<struct<"class.sycl::_V1::nd_range.3", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>>) -> !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr<struct<"class.sycl::_V1::nd_range.3", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>>) -> !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
// CHECK:           llvm.return %[[VAL_2]] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK:         }
func.func @test(%mr: memref<?x!sycl_nd_range_3_>) -> !sycl_range_3_ {
  %0 = "sycl.nd_range.get_global_range"(%mr) { ArgumentTypes = [memref<?x!sycl_nd_range_3_>], FunctionName = @"get_global_range", MangledFunctionName = @"get_global_range", TypeName = @"nd_range" }  : (memref<?x!sycl_nd_range_3_>) -> !sycl_range_3_
  return %0 : !sycl_range_3_
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.nd_range.get_local_range
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_nd_range_3_ = !sycl.nd_range<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<struct<"class.sycl::_V1::nd_range.3", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>>) -> !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr<struct<"class.sycl::_V1::nd_range.3", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>>) -> !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
// CHECK:           llvm.return %[[VAL_2]] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK:         }
func.func @test(%mr: memref<?x!sycl_nd_range_3_>) -> !sycl_range_3_ {
  %0 = "sycl.nd_range.get_local_range"(%mr) { ArgumentTypes = [memref<?x!sycl_nd_range_3_>], FunctionName = @"get_local_range", MangledFunctionName = @"get_local_range", TypeName = @"nd_range" }  : (memref<?x!sycl_nd_range_3_>) -> !sycl_range_3_
  return %0 : !sycl_range_3_
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.nd_range.get_group_range
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_nd_range_3_ = !sycl.nd_range<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<struct<"class.sycl::_V1::nd_range.3", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>>) -> !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> : (i64) -> !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0] : (!llvm.ptr<struct<"class.sycl::_V1::nd_range.3", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 0] : (!llvm.ptr<struct<"class.sycl::_V1::nd_range.3", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_7:.*]] = llvm.udiv %[[VAL_4]], %[[VAL_6]]  : i64
// CHECK:           %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0, 0, 0] : (!llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>) -> !llvm.ptr<i64>
// CHECK:           llvm.store %[[VAL_7]], %[[VAL_8]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 1] : (!llvm.ptr<struct<"class.sycl::_V1::nd_range.3", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 1] : (!llvm.ptr<struct<"class.sycl::_V1::nd_range.3", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_13:.*]] = llvm.udiv %[[VAL_10]], %[[VAL_12]]  : i64
// CHECK:           %[[VAL_14:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0, 0, 1] : (!llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>) -> !llvm.ptr<i64>
// CHECK:           llvm.store %[[VAL_13]], %[[VAL_14]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_15:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 2] : (!llvm.ptr<struct<"class.sycl::_V1::nd_range.3", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_16:.*]] = llvm.load %[[VAL_15]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_17:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 2] : (!llvm.ptr<struct<"class.sycl::_V1::nd_range.3", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_18:.*]] = llvm.load %[[VAL_17]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_19:.*]] = llvm.udiv %[[VAL_16]], %[[VAL_18]]  : i64
// CHECK:           %[[VAL_20:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0, 0, 2] : (!llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>) -> !llvm.ptr<i64>
// CHECK:           llvm.store %[[VAL_19]], %[[VAL_20]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_21:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
// CHECK:           llvm.return %[[VAL_21]] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK:         }
func.func @test(%mr: memref<?x!sycl_nd_range_3_>) -> !sycl_range_3_ {
  %0 = "sycl.nd_range.get_group_range"(%mr) { ArgumentTypes = [memref<?x!sycl_nd_range_3_>], FunctionName = @"get_group_range", MangledFunctionName = @"get_group_range", TypeName = @"nd_range" }  : (memref<?x!sycl_nd_range_3_>) -> !sycl_range_3_
  return %0 : !sycl_range_3_
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.range.get with scalar result type
//===-------------------------------------------------------------------------------------------------===//

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK:           %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, %[[VAL_1]]] : (!llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>, i32) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK:           llvm.return %[[VAL_3]] : i64
// CHECK:         }
func.func @test(%mr: memref<?x!sycl_range_3_>, %idx: i32) -> i64 {
  %0 = "sycl.range.get"(%mr, %idx) { ArgumentTypes = [memref<?x!sycl_range_3_>, i32], FunctionName = @"get", MangledFunctionName = @"get", TypeName = @"range" }  : (memref<?x!sycl_range_3_>, i32) -> i64
  return %0 : i64
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.range.get with reference result type
//===-------------------------------------------------------------------------------------------------===//

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[RANGE3:.*]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> !llvm.ptr<i64> {
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[RANGE3]]>, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:      llvm.return %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:    }
func.func @test(%mr: memref<?x!sycl_range_3_>, %idx: i32) -> memref<?xi64> {
  %0 = "sycl.range.get"(%mr, %idx) { ArgumentTypes = [memref<?x!sycl_range_3_>, i32], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"range" }  : (memref<?x!sycl_range_3_>, i32) -> memref<?xi64>
  return %0 : memref<?xi64>
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.range.size
//===-------------------------------------------------------------------------------------------------===//

!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<3xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<3xi64, 4>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

// CHECK-LABEL:   llvm.func @test1(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>,
// CHECK-SAME:                     %[[VAL_1:.*]]: i32) -> i64 {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0] : (!llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK:           llvm.return %[[VAL_5]] : i64
// CHECK:         }
func.func @test1(%mr: memref<?x!sycl_range_1_>, %idx: i32) -> i64 {
  %0 = "sycl.range.size"(%mr) { ArgumentTypes = [memref<?x!sycl_range_1_>], FunctionName = @"size", MangledFunctionName = @"size", TypeName = @"range" }  : (memref<?x!sycl_range_1_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test2(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>,
// CHECK-SAME:                     %[[VAL_1:.*]]: i32) -> i64 {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0] : (!llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK:           %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1] : (!llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_8:.*]] = llvm.mul %[[VAL_5]], %[[VAL_7]]  : i64
// CHECK:           llvm.return %[[VAL_8]] : i64
// CHECK:         }
func.func @test2(%mr: memref<?x!sycl_range_2_>, %idx: i32) -> i64 {
  %0 = "sycl.range.size"(%mr) { ArgumentTypes = [memref<?x!sycl_range_2_>], FunctionName = @"size", MangledFunctionName = @"size", TypeName = @"range" }  : (memref<?x!sycl_range_2_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test3(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>,
// CHECK-SAME:                     %[[VAL_1:.*]]: i32) -> i64 {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0] : (!llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK:           %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1] : (!llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_8:.*]] = llvm.mul %[[VAL_5]], %[[VAL_7]]  : i64
// CHECK:           %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 2] : (!llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_11:.*]] = llvm.mul %[[VAL_8]], %[[VAL_10]]  : i64
// CHECK:           llvm.return %[[VAL_11]] : i64
// CHECK:         }
func.func @test3(%mr: memref<?x!sycl_range_3_>, %idx: i32) -> i64 {
  %0 = "sycl.range.size"(%mr) { ArgumentTypes = [memref<?x!sycl_range_3_>], FunctionName = @"size", MangledFunctionName = @"size", TypeName = @"range" }  : (memref<?x!sycl_range_3_>) -> i64
  return %0 : i64
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.accessor.subscript with scalar offset
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR1:.*]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i64) -> !llvm.ptr<i32, 1> {
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, %[[VAL_1]]] : (!llvm.ptr<[[ACCESSOR1]]>, i64) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:      llvm.return %[[VAL_5]] : !llvm.ptr<i32, 1>
// CHECK-NEXT:    }
func.func @test(%mr: memref<?x!sycl_accessor_1_i32_rw_gb>, %off: i64) -> (memref<?xi32, 1>) {
  %0 = sycl.accessor.subscript %mr[%off] { ArgumentTypes = [memref<?x!sycl_accessor_1_i32_rw_gb>, i64], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"accessor" }  : (memref<?x!sycl_accessor_1_i32_rw_gb>, i64) -> memref<?xi32, 1>
  return %0 : memref<?xi32, 1>
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.accessor.subscript with sycl.id offset
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_accessor_2_i32_rw_gb = !sycl.accessor<[2, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_accessor_3_i32_rw_gb = !sycl.accessor<[3, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[3], (!sycl_id_3_, !sycl_range_3_, !sycl_range_3_)>, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK-LABEL:   llvm.func @test0(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR1]]>,
// CHECK-SAME:                     %[[VAL_1:.*]]: !llvm.ptr<[[ID1:.*]]>) -> !llvm.ptr<i32, 1> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ACCESSOR1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr<[[ID1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.load %[[VAL_10]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.mul %[[VAL_2]], %[[VAL_9]]  : i64
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.add %[[VAL_12]], %[[VAL_11]]  : i64
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, %[[VAL_13]]] : (!llvm.ptr<[[ACCESSOR1]]>, i64) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:      llvm.return %[[VAL_14]] : !llvm.ptr<i32, 1>
// CHECK-NEXT:    }
func.func @test0(%mr: memref<?x!sycl_accessor_1_i32_rw_gb>, %off_mr: memref<?x!sycl_id_1_>) -> (memref<?xi32, 1>) {
  %0 = sycl.accessor.subscript %mr[%off_mr] { ArgumentTypes = [memref<?x!sycl_accessor_1_i32_rw_gb>, memref<?x!sycl_id_1_>], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"accessor" }  : (memref<?x!sycl_accessor_1_i32_rw_gb>, memref<?x!sycl_id_1_>) -> memref<?xi32, 1>
  return %0 : memref<?xi32, 1>
}

// CHECK-LABEL:   llvm.func @test1(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR2:.*]]>,
// CHECK-SAME:                     %[[VAL_1:.*]]: !llvm.ptr<[[ID2:.*]]>) -> !llvm.ptr<i32, 1> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ACCESSOR2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr<[[ID2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.add %[[VAL_7]], %[[VAL_6]]  : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 1] : (!llvm.ptr<[[ACCESSOR2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 1] : (!llvm.ptr<[[ID2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.mul %[[VAL_8]], %[[VAL_10]]  : i64
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.add %[[VAL_13]], %[[VAL_12]]  : i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, %[[VAL_14]]] : (!llvm.ptr<[[ACCESSOR2]]>, i64) -> !llvm.ptr<i32, 1>
func.func @test1(%mr: memref<?x!sycl_accessor_2_i32_rw_gb>, %off_mr: memref<?x!sycl_id_2_>) -> (memref<?xi32, 1>) {
  %0 = sycl.accessor.subscript %mr[%off_mr] { ArgumentTypes = [memref<?x!sycl_accessor_2_i32_rw_gb>, memref<?x!sycl_id_2_>], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"accessor" }  : (memref<?x!sycl_accessor_2_i32_rw_gb>, memref<?x!sycl_id_2_>) -> memref<?xi32, 1>
  return %0 : memref<?xi32, 1>
}

// CHECK-LABEL:   llvm.func @test2(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR3:.*]]>,
// CHECK-SAME:                     %[[VAL_1:.*]]: !llvm.ptr<[[ID3:.*]]>) -> !llvm.ptr<i32, 1> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ACCESSOR3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr<[[ID3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.add %[[VAL_7]], %[[VAL_6]]  : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 1] : (!llvm.ptr<[[ACCESSOR3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 1] : (!llvm.ptr<[[ID3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.mul %[[VAL_8]], %[[VAL_10]]  : i64
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.add %[[VAL_13]], %[[VAL_12]]  : i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 2] : (!llvm.ptr<[[ACCESSOR3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_16:.*]] = llvm.load %[[VAL_15]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_17:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 2] : (!llvm.ptr<[[ID3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_18:.*]] = llvm.load %[[VAL_17]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_19:.*]] = llvm.mul %[[VAL_14]], %[[VAL_16]]  : i64
// CHECK-NEXT:      %[[VAL_20:.*]] = llvm.add %[[VAL_19]], %[[VAL_18]]  : i64
// CHECK-NEXT:      %[[VAL_21:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, %[[VAL_20]]] : (!llvm.ptr<[[ACCESSOR3]]>, i64) -> !llvm.ptr<i32, 1>
func.func @test2(%mr: memref<?x!sycl_accessor_3_i32_rw_gb>, %off_mr: memref<?x!sycl_id_3_>) -> (memref<?xi32, 1>) {
  %0 = sycl.accessor.subscript %mr[%off_mr] { ArgumentTypes = [memref<?x!sycl_accessor_3_i32_rw_gb>, memref<?x!sycl_id_3_>], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"accessor" }  : (memref<?x!sycl_accessor_3_i32_rw_gb>, memref<?x!sycl_id_3_>) -> memref<?xi32, 1>
  return %0 : memref<?xi32, 1>
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.accessor.subscript with ND accessor and scalar offset
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_accessor_impl_device_3_ = !sycl.accessor_impl_device<[3], (!sycl_id_3_, !sycl_range_3_, !sycl_range_3_)>
!sycl_accessor_3_i32_rw_gb = !sycl.accessor<[3, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[3], (!sycl_id_3_, !sycl_range_3_, !sycl_range_3_)>, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_accessor_subscript_2_ = !sycl.accessor_subscript<[2], (!sycl_id_2_, !sycl.accessor<[3, i32, read_write, global_buffer], (!sycl_accessor_impl_device_3_, !llvm.struct<(ptr<i32, 1>)>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR3]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i64) -> !llvm.[[SUBSCRIPT2:.*]] {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.undef : !llvm.[[SUBSCRIPT2]]
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_2]][0, 0, 0, 0] : !llvm.[[SUBSCRIPT2]]
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_3]][0, 0, 0, 1] : !llvm.[[SUBSCRIPT2]]
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_0]] : !llvm.ptr<[[ACCESSOR3]]>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_5]][1] : !llvm.[[SUBSCRIPT2]]
// CHECK-NEXT:      llvm.return %[[VAL_7]] : !llvm.[[SUBSCRIPT2]]
// CHECK-NEXT:    }
func.func @test(%mr: memref<?x!sycl_accessor_3_i32_rw_gb>, %off: i64) -> !sycl_accessor_subscript_2_ {
  %0 = sycl.accessor.subscript %mr[%off] { ArgumentTypes = [memref<?x!sycl_accessor_3_i32_rw_gb>, i64], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"accessor" }  : (memref<?x!sycl_accessor_3_i32_rw_gb>, i64) -> !sycl_accessor_subscript_2_
  return %0 : !sycl_accessor_subscript_2_
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.accessor.subscript with atomic access mode and scalar offset
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_accessor_impl_device_3_ = !sycl.accessor_impl_device<[3], (!sycl_id_3_, !sycl_range_3_, !sycl_range_3_)>
!sycl_accessor_3_i32_ato_gb = !sycl.accessor<[3, i32, atomic, global_buffer], (!sycl.accessor_impl_device<[3], (!sycl_id_3_, !sycl_range_3_, !sycl_range_3_)>, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_accessor_subscript_2_ = !sycl.accessor_subscript<[2], (!sycl_id_2_, !sycl.accessor<[3, i32, atomic, global_buffer], (!sycl_accessor_impl_device_3_, !llvm.struct<(ptr<i32, 1>)>)>)>
!sycl_atomic_i32_3_ = !sycl.atomic<[i32, 1], (memref<?xi32, 1>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<struct<"class.sycl::_V1::accessor.3", (struct<"class.sycl::_V1::detail::AccessorImplDevice.3", (struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>, struct<(ptr<i32, 1>)>)>>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i64) -> !llvm.struct<"class.sycl::_V1::atomic", (ptr<i32, 1>)> {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.undef : !llvm.struct<"class.sycl::_V1::atomic", (ptr<i32, 1>)>
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, %[[VAL_1]]] : (!llvm.ptr<struct<"class.sycl::_V1::accessor.3", (struct<"class.sycl::_V1::detail::AccessorImplDevice.3", (struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>, struct<(ptr<i32, 1>)>)>>, i64) -> !llvm.ptr<i32, 1>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_2]][0] : !llvm.struct<"class.sycl::_V1::atomic", (ptr<i32, 1>)>
// CHECK:           llvm.return %[[VAL_4]] : !llvm.struct<"class.sycl::_V1::atomic", (ptr<i32, 1>)>
// CHECK:         }
func.func @test(%mr: memref<?x!sycl_accessor_3_i32_ato_gb>, %off: i64) -> !sycl_atomic_i32_3_ {
  %0 = sycl.accessor.subscript %mr[%off] { ArgumentTypes = [memref<?x!sycl_accessor_3_i32_ato_gb>, i64], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"accessor" }  : (memref<?x!sycl_accessor_3_i32_ato_gb>, i64) -> !sycl_atomic_i32_3_
  return %0 : !sycl_atomic_i32_3_
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.item.get_linear_id with offset
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>)>

// CHECK-LABEL:   llvm.func @test1(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr<[[ITEMOFF1:.*]]>) -> i64 {
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ITEMOFF1]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr<[[ITEMOFF1]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_5:.*]] = llvm.sub %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK:           llvm.return %[[VAL_5]] : i64
// CHECK:         }
func.func @test1(%mr: memref<?x!sycl_item_1_>) -> i64 {
  %0 = "sycl.item.get_linear_id"(%mr) { ArgumentTypes = [memref<?x!sycl_item_1_>], FunctionName = @"get_linear_id", MangledFunctionName = @"get_linear_id", TypeName = @"item" }  : (memref<?x!sycl_item_1_>) -> i64
  return %0 : i64
}

!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_item_2_ = !sycl.item<[2, true], (!sycl.item_base<[2, true], (!sycl_range_2_, !sycl_id_2_, !sycl_id_2_)>)>

// CHECK-LABEL:   llvm.func @test2(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr<[[ITEMOFF2:.*]]>) -> i64 {
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ITEMOFF2]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr<[[ITEMOFF2]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_5:.*]] = llvm.sub %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK:           %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 1] : (!llvm.ptr<[[ITEMOFF2]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_8:.*]] = llvm.mul %[[VAL_5]], %[[VAL_7]]  : i64
// CHECK:           %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 1] : (!llvm.ptr<[[ITEMOFF2]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 1] : (!llvm.ptr<[[ITEMOFF2]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_13:.*]] = llvm.sub %[[VAL_10]], %[[VAL_12]]  : i64
// CHECK:           %[[VAL_14:.*]] = llvm.add %[[VAL_8]], %[[VAL_13]]  : i64
// CHECK:           llvm.return %[[VAL_14]] : i64
// CHECK:         }
func.func @test2(%mr: memref<?x!sycl_item_2_>) -> i64 {
  %0 = "sycl.item.get_linear_id"(%mr) { ArgumentTypes = [memref<?x!sycl_item_2_>], FunctionName = @"get_linear_id", MangledFunctionName = @"get_linear_id", TypeName = @"item" }  : (memref<?x!sycl_item_2_>) -> i64
  return %0 : i64
}

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_item_3_ = !sycl.item<[3, true], (!sycl.item_base<[3, true], (!sycl_range_3_, !sycl_id_3_, !sycl_id_3_)>)>

// CHECK-LABEL:   llvm.func @test3(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr<[[ITEMOFF3:.*]]>) -> i64 {
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ITEMOFF3]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr<[[ITEMOFF3]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_5:.*]] = llvm.sub %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK:           %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 1] : (!llvm.ptr<[[ITEMOFF3]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_8:.*]] = llvm.mul %[[VAL_5]], %[[VAL_7]]  : i64
// CHECK:           %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 2] : (!llvm.ptr<[[ITEMOFF3]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_11:.*]] = llvm.mul %[[VAL_8]], %[[VAL_10]]  : i64
// CHECK:           %[[VAL_12:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 1] : (!llvm.ptr<[[ITEMOFF3]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_13:.*]] = llvm.load %[[VAL_12]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_14:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 1] : (!llvm.ptr<[[ITEMOFF3]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_15:.*]] = llvm.load %[[VAL_14]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_16:.*]] = llvm.sub %[[VAL_13]], %[[VAL_15]]  : i64
// CHECK:           %[[VAL_17:.*]] = llvm.mul %[[VAL_16]], %[[VAL_10]]  : i64
// CHECK:           %[[VAL_18:.*]] = llvm.add %[[VAL_11]], %[[VAL_17]]  : i64
// CHECK:           %[[VAL_19:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 2] : (!llvm.ptr<[[ITEMOFF3]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_20:.*]] = llvm.load %[[VAL_19]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_21:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 2] : (!llvm.ptr<[[ITEMOFF3]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_22:.*]] = llvm.load %[[VAL_21]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_23:.*]] = llvm.sub %[[VAL_20]], %[[VAL_22]]  : i64
// CHECK:           %[[VAL_24:.*]] = llvm.add %[[VAL_18]], %[[VAL_23]]  : i64
// CHECK:           llvm.return %[[VAL_24]] : i64
// CHECK:         }
func.func @test3(%mr: memref<?x!sycl_item_3_>) -> i64 {
  %0 = "sycl.item.get_linear_id"(%mr) { ArgumentTypes = [memref<?x!sycl_item_3_>], FunctionName = @"get_linear_id", MangledFunctionName = @"get_linear_id", TypeName = @"item" }  : (memref<?x!sycl_item_3_>) -> i64
  return %0 : i64
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.item.get_id with id output
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_item_2_ = !sycl.item<[2, false], (!sycl.item_base<[2, false], (!sycl_range_2_, !sycl_id_2_)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ITEM2:.*]]>) -> !llvm.[[ID2]] {
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1] : (!llvm.ptr<[[ITEM2]]>) -> !llvm.ptr<[[ID2]]>
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<[[ID2]]>
// CHECK:           llvm.return %[[VAL_2]] : !llvm.[[ID2]]
// CHECK:         }
func.func @test(%mr: memref<?x!sycl_item_2_>) -> !sycl_id_2_ {
  %0 = "sycl.item.get_id"(%mr) { ArgumentTypes = [memref<?x!sycl_item_2_>], FunctionName = @"get_id", MangledFunctionName = @"get_id", TypeName = @"item" }  : (memref<?x!sycl_item_2_>) -> !sycl_id_2_
  return %0 : !sycl_id_2_
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.item.get_id with i64 output
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_item_2_ = !sycl.item<[2, false], (!sycl.item_base<[2, false], (!sycl_range_2_, !sycl_id_2_)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ITEM2]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK:           %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[ITEM2]]>, i32) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK:           llvm.return %[[VAL_3]] : i64
// CHECK:         }
func.func @test(%mr: memref<?x!sycl_item_2_>, %index: i32) -> i64 {
   %0 = "sycl.item.get_id"(%mr, %index) { ArgumentTypes = [memref<?x!sycl_item_2_>, i32], FunctionName = @"get_id", MangledFunctionName = @"get_id", TypeName = @"item" }  : (memref<?x!sycl_item_2_>, i32) -> i64
  return %0 : i64
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.item.get_range with i64 output
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_item_2_ = !sycl.item<[2, false], (!sycl.item_base<[2, false], (!sycl_range_2_, !sycl_id_2_)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ITEM2]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK:           %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[ITEM2]]>, i32) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK:           llvm.return %[[VAL_3]] : i64
// CHECK:         }
func.func @test(%mr: memref<?x!sycl_item_2_>, %index: i32) -> i64 {
   %0 = "sycl.item.get_range"(%mr, %index) { ArgumentTypes = [memref<?x!sycl_item_2_>, i32], FunctionName = @"get_range", MangledFunctionName = @"get_range", TypeName = @"item" }  : (memref<?x!sycl_item_2_>, i32) -> i64
  return %0 : i64
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.item.get_range with range output
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_item_2_ = !sycl.item<[2, false], (!sycl.item_base<[2, false], (!sycl_range_2_, !sycl_id_2_)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ITEM2]]>) -> !llvm.[[RANGE2:.*]] {
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0] : (!llvm.ptr<[[ITEM2]]>) -> !llvm.ptr<[[RANGE2]]>
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<[[RANGE2]]>
// CHECK:           llvm.return %[[VAL_2]] : !llvm.[[RANGE2]]
// CHECK:         }
func.func @test(%mr: memref<?x!sycl_item_2_>) -> !sycl_range_2_ {
  %0 = "sycl.item.get_range"(%mr) { ArgumentTypes = [memref<?x!sycl_item_2_>], FunctionName = @"get_range", MangledFunctionName = @"get_range", TypeName = @"item" }  : (memref<?x!sycl_item_2_>) -> !sycl_range_2_
  return %0 : !sycl_range_2_
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.item.get_linear_id no offset
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_item_1_ = !sycl.item<[1, false], (!sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>)>

// CHECK-LABEL:   llvm.func @test1(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr<[[ITEM1:.*]]>) -> i64 {
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ITEM1]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK:           llvm.return %[[VAL_2]] : i64
// CHECK:         }
func.func @test1(%mr: memref<?x!sycl_item_1_>) -> i64 {
  %0 = "sycl.item.get_linear_id"(%mr) { ArgumentTypes = [memref<?x!sycl_item_1_>], FunctionName = @"get_linear_id", MangledFunctionName = @"get_linear_id", TypeName = @"item" }  : (memref<?x!sycl_item_1_>) -> i64
  return %0 : i64
}

!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_item_2_ = !sycl.item<[2, false], (!sycl.item_base<[2, false], (!sycl_range_2_, !sycl_id_2_)>)>

// CHECK-LABEL:   llvm.func @test2(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr<[[ITEM2]]>) -> i64 {
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ITEM2]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 1] : (!llvm.ptr<[[ITEM2]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK:           %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 1] : (!llvm.ptr<[[ITEM2]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_8:.*]] = llvm.add %[[VAL_5]], %[[VAL_7]]  : i64
// CHECK:           llvm.return %[[VAL_8]] : i64
// CHECK:         }
func.func @test2(%mr: memref<?x!sycl_item_2_>) -> i64 {
  %0 = "sycl.item.get_linear_id"(%mr) { ArgumentTypes = [memref<?x!sycl_item_2_>], FunctionName = @"get_linear_id", MangledFunctionName = @"get_linear_id", TypeName = @"item" }  : (memref<?x!sycl_item_2_>) -> i64
  return %0 : i64
}

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_item_3_ = !sycl.item<[3, false], (!sycl.item_base<[3, false], (!sycl_range_3_, !sycl_id_3_)>)>

// CHECK-LABEL:   llvm.func @test3(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr<[[ITEM3:.*]]>) -> i64 {
// CHECK:           %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ITEM3]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 1] : (!llvm.ptr<[[ITEM3]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK:           %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 2] : (!llvm.ptr<[[ITEM3]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_8:.*]] = llvm.mul %[[VAL_5]], %[[VAL_7]]  : i64
// CHECK:           %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 1] : (!llvm.ptr<[[ITEM3]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_11:.*]] = llvm.mul %[[VAL_10]], %[[VAL_7]]  : i64
// CHECK:           %[[VAL_12:.*]] = llvm.add %[[VAL_8]], %[[VAL_11]]  : i64
// CHECK:           %[[VAL_13:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 2] : (!llvm.ptr<[[ITEM3]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_15:.*]] = llvm.add %[[VAL_12]], %[[VAL_14]]  : i64
// CHECK:           llvm.return %[[VAL_15]] : i64
// CHECK:         }
func.func @test3(%mr: memref<?x!sycl_item_3_>) -> i64 {
  %0 = "sycl.item.get_linear_id"(%mr) { ArgumentTypes = [memref<?x!sycl_item_3_>], FunctionName = @"get_linear_id", MangledFunctionName = @"get_linear_id", TypeName = @"item" }  : (memref<?x!sycl_item_3_>) -> i64
  return %0 : i64
}
