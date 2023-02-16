// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm="use-bare-ptr-call-conv" %s | FileCheck %s
// XFAIL: *

//===-------------------------------------------------------------------------------------------------===//
// sycl.nd_range.get_global_range
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_nd_range_3_ = !sycl.nd_range<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.struct<"class.sycl::_V1::nd_range.3", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>) -> !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<"class.sycl::_V1::nd_range.3", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>
// CHECK-NEXT:      llvm.return %[[VAL_1]] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:    }
func.func @test(%nd_range: !sycl_nd_range_3_) -> !sycl_range_3_ {
  %0 = "sycl.nd_range.get_global_range"(%nd_range) { ArgumentTypes = [!sycl_nd_range_3_], FunctionName = @"get_global_range", MangledFunctionName = @"get_global_range", TypeName = @"nd_range" }  : (!sycl_nd_range_3_) -> !sycl_range_3_
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
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.struct<"class.sycl::_V1::nd_range.3", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>) -> !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.extractvalue %[[VAL_0]][1] : !llvm.struct<"class.sycl::_V1::nd_range.3", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>
// CHECK-NEXT:      llvm.return %[[VAL_1]] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:    }
func.func @test(%nd_range: !sycl_nd_range_3_) -> !sycl_range_3_ {
  %0 = "sycl.nd_range.get_local_range"(%nd_range) { ArgumentTypes = [!sycl_nd_range_3_], FunctionName = @"get_local_range", MangledFunctionName = @"get_local_range", TypeName = @"nd_range" }  : (!sycl_nd_range_3_) -> !sycl_range_3_
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
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.struct<"class.sycl::_V1::nd_range.3", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>) -> !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<"class.sycl::_V1::nd_range.3", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.extractvalue %[[VAL_0]][1] : !llvm.struct<"class.sycl::_V1::nd_range.3", (struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>)>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.undef : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.extractvalue %[[VAL_1]][0, 0, 0] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.extractvalue %[[VAL_2]][0, 0, 0] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.udiv %[[VAL_4]], %[[VAL_5]]  : i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_3]][0, 0, 0] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.extractvalue %[[VAL_1]][0, 0, 1] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.extractvalue %[[VAL_2]][0, 0, 1] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.udiv %[[VAL_8]], %[[VAL_9]]  : i64
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_10]], %[[VAL_7]][0, 0, 1] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.extractvalue %[[VAL_1]][0, 0, 2] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.extractvalue %[[VAL_2]][0, 0, 2] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.udiv %[[VAL_12]], %[[VAL_13]]  : i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.insertvalue %[[VAL_14]], %[[VAL_11]][0, 0, 2] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      llvm.return %[[VAL_15]] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:    }
func.func @test(%nd_range: !sycl_nd_range_3_) -> !sycl_range_3_ {
  %0 = "sycl.nd_range.get_group_range"(%nd_range) { ArgumentTypes = [!sycl_nd_range_3_], FunctionName = @"get_group_range", MangledFunctionName = @"get_group_range", TypeName = @"nd_range" }  : (!sycl_nd_range_3_) -> !sycl_range_3_
  return %0 : !sycl_range_3_
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.range.get with scalar result type
//===-------------------------------------------------------------------------------------------------===//

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {alignment = 24 : i64} : (i64) -> !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
// CHECK-NEXT:      llvm.store %[[VAL_0]], %[[VAL_3]] : !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_3]][0, 0, 0, %[[VAL_1]]] : (!llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.load %[[VAL_4]] {alignment = 24 : i64} : !llvm.ptr<i64>
// CHECK-NEXT:      llvm.return %[[VAL_5]] : i64
// CHECK-NEXT:    }
func.func @test(%range: !sycl_range_3_, %idx: i32) -> i64 {
  %0 = "sycl.range.get"(%range, %idx) { ArgumentTypes = [!sycl_range_3_, i32], FunctionName = @"get", MangledFunctionName = @"get", TypeName = @"range" }  : (!sycl_range_3_, i32) -> i64
  return %0 : i64
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.range.size
//===-------------------------------------------------------------------------------------------------===//

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.extractvalue %[[VAL_0]][0, 0, 0] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mul %[[VAL_1]], %[[VAL_2]]  : i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.extractvalue %[[VAL_0]][0, 0, 1] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.mul %[[VAL_3]], %[[VAL_4]]  : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.extractvalue %[[VAL_0]][0, 0, 2] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_5]], %[[VAL_6]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_7]] : i64
// CHECK-NEXT:    }
func.func @test(%range: !sycl_range_3_) -> i64 {
  %0 = "sycl.range.size"(%range) { ArgumentTypes = [!sycl_range_3_], FunctionName = @"size", MangledFunctionName = @"size", TypeName = @"range" }  : (!sycl_range_3_) -> i64
  return %0 : i64
}
