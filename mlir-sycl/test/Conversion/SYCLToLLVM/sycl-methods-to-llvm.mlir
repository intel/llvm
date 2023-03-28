// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm %s | FileCheck %s

//===-------------------------------------------------------------------------------------------------===//
// sycl.range.get with scalar result type
//===-------------------------------------------------------------------------------------------------===//

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>

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

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>

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

!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>

// CHECK-LABEL:   llvm.func @test_1(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[RANGE1:.*]]>) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0] : (!llvm.ptr<[[RANGE1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mul %[[VAL_1]], %[[VAL_3]] : i64
// CHECK-NEXT:      llvm.return %[[VAL_4]] : i64
// CHECK-NEXT:    }
func.func @test_1(%range: memref<?x!sycl_range_1_>) -> i64 {
  %0 = sycl.range.size(%range) { ArgumentTypes = [memref<?x!sycl_range_1_>], FunctionName = @"size", MangledFunctionName = @"size", TypeName = @"range" }  : (memref<?x!sycl_range_1_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[RANGE2:.*]]>) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0] : (!llvm.ptr<[[RANGE2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mul %[[VAL_1]], %[[VAL_3]] : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1] : (!llvm.ptr<[[RANGE2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_4]], %[[VAL_6]] : i64
// CHECK-NEXT:      llvm.return %[[VAL_7]] : i64
// CHECK-NEXT:    }
func.func @test_2(%range: memref<?x!sycl_range_2_>) -> i64 {
  %0 = sycl.range.size(%range) { ArgumentTypes = [memref<?x!sycl_range_2_>], FunctionName = @"size", MangledFunctionName = @"size", TypeName = @"range" }  : (memref<?x!sycl_range_2_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[RANGE3]]>) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0] : (!llvm.ptr<[[RANGE3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mul %[[VAL_1]], %[[VAL_3]] : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1] : (!llvm.ptr<[[RANGE3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_4]], %[[VAL_6]] : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 2] : (!llvm.ptr<[[RANGE3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.mul %[[VAL_7]], %[[VAL_9]] : i64
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

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>

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
// sycl.id.get with scalar result type and no argument
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ID1:.*]]>) -> i64 {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[ID1]]>, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:      llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:    }
func.func @test(%id: memref<?x!sycl_id_1_>) -> i64 {
  %0 = sycl.id.get %id[] { ArgumentTypes = [memref<?x!sycl_id_1_>], FunctionName = @"operator unsigned long", MangledFunctionName = @"operator unsigned long", TypeName = @"id" }  : (memref<?x!sycl_id_1_>) -> i64
  return %0 : i64
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.id.get with reference result type
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>

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
// sycl.accessor.get_pointer
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR1:.*]]>) -> !llvm.ptr<i32, 1> {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr<[[ACCESSOR1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 0] : (!llvm.ptr<[[ACCESSOR1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]]  : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.add %[[VAL_7]], %[[VAL_6]]  : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.sub %[[VAL_1]], %[[VAL_8]] : i64
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0] : (!llvm.ptr<[[ACCESSOR1]]>) -> !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.load %[[VAL_10]] : !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.getelementptr inbounds %[[VAL_11]][%[[VAL_9]]] : (!llvm.ptr<i32, 1>, i64) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:      llvm.return %[[VAL_12]] : !llvm.ptr<i32, 1>
// CHECK-NEXT:    }
func.func @test(%acc: memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?xi32, 1> {
  %0 = sycl.accessor.get_pointer(%acc) { ArgumentTypes = [memref<?x!sycl_accessor_1_i32_rw_gb>], FunctionName = @"get_pointer", MangledFunctionName = @"get_pointer", TypeName = @"accessor" }  : (memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?xi32, 1>
  return %0 : memref<?xi32, 1>
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.accessor.get_range
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR1:.*]]>) -> !llvm.[[RANGE1:.*]] {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1] : (!llvm.ptr<[[ACCESSOR1]]>) -> !llvm.ptr<[[RANGE1]]>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<[[RANGE1]]>
// CHECK-NEXT:      llvm.return %[[VAL_2]] : !llvm.[[RANGE1]]
// CHECK-NEXT:    }
func.func @test(%acc: memref<?x!sycl_accessor_1_i32_rw_gb>) -> !sycl_range_1_ {
  %0 = sycl.accessor.get_range(%acc) { ArgumentTypes = [memref<?x!sycl_accessor_1_i32_rw_gb>], FunctionName = @"get_range", MangledFunctionName = @"get_range", TypeName = @"accessor" }  : (memref<?x!sycl_accessor_1_i32_rw_gb>) -> !sycl_range_1_
  return %0 : !sycl_range_1_
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.accessor.size
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR1:.*]]>) -> i64 {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ACCESSOR1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mul %[[VAL_1]], %[[VAL_3]]  : i64
// CHECK-NEXT:      llvm.return %[[VAL_4]] : i64
// CHECK-NEXT:    }
func.func @test(%acc: memref<?x!sycl_accessor_1_i32_rw_gb>) -> i64 {
  %0 = sycl.accessor.size(%acc) { ArgumentTypes = [memref<?x!sycl_accessor_1_i32_rw_gb>], FunctionName = @"size", MangledFunctionName = @"size", TypeName = @"accessor" }  : (memref<?x!sycl_accessor_1_i32_rw_gb>) -> i64
  return %0 : i64
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// sycl.accessor.subscript with scalar offset and 1D accessor
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
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
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR2:.*]]>, %[[VAL_1:.*]]: i64) -> !llvm.[[ACCESSORSUBS2:.*]] {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.undef : !llvm.[[ACCESSORSUBS2]]
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_2]][0, 0, 0, 0] : !llvm.[[ACCESSORSUBS2]]
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mlir.constant(0 : i64) : i64
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
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mlir.constant(0 : i64) : i64
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
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR1]]>,
// CHECK-SAME:                      %[[VAL_1:.*]]: !llvm.ptr<[[ID1]]>) -> !llvm.ptr<i32, 1> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr<[[ACCESSOR1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr<[[ID1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.add %[[VAL_7]], %[[VAL_6]] : i64
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
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr<[[ACCESSOR2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr<[[ID2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.add %[[VAL_7]], %[[VAL_6]] : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 1] : (!llvm.ptr<[[ACCESSOR2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 1] : (!llvm.ptr<[[ID2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.mul %[[VAL_8]], %[[VAL_10]] : i64
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.add %[[VAL_13]], %[[VAL_12]] : i64
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
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr<[[ACCESSOR3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr<[[ID3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.add %[[VAL_7]], %[[VAL_6]] : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 1] : (!llvm.ptr<[[ACCESSOR3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 1] : (!llvm.ptr<[[ID3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.mul %[[VAL_8]], %[[VAL_10]] : i64
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.add %[[VAL_13]], %[[VAL_12]] : i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 2] : (!llvm.ptr<[[ACCESSOR3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_16:.*]] = llvm.load %[[VAL_15]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_17:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 2] : (!llvm.ptr<[[ID3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_18:.*]] = llvm.load %[[VAL_17]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_19:.*]] = llvm.mul %[[VAL_14]], %[[VAL_16]] : i64
// CHECK-NEXT:      %[[VAL_20:.*]] = llvm.add %[[VAL_19]], %[[VAL_18]] : i64
// CHECK-NEXT:      %[[VAL_21:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0] : (!llvm.ptr<[[ACCESSOR3]]>) -> !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_22:.*]] = llvm.load %[[VAL_21]] : !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %[[VAL_23:.*]] = llvm.getelementptr inbounds %[[VAL_22]][%[[VAL_20]]] : (!llvm.ptr<i32, 1>, i64) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:      llvm.return %[[VAL_23]] : !llvm.ptr<i32, 1>
// CHECK-NEXT:    }
func.func @test_3(%acc: memref<?x!sycl_accessor_3_i32_rw_gb>, %idx: memref<?x!sycl_id_3_>) -> memref<?xi32, 1> {
  %0 = sycl.accessor.subscript %acc[%idx] { ArgumentTypes = [memref<?x!sycl_accessor_3_i32_rw_gb>, memref<?x!sycl_id_3_>], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"id" }  : (memref<?x!sycl_accessor_3_i32_rw_gb>, memref<?x!sycl_id_3_>) -> memref<?xi32, 1>
  return %0 : memref<?xi32, 1>
}

// CHECK-LABEL:   llvm.func @test_struct(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSORSTR:.*]]>, %[[VAL_1:.*]]: !llvm.ptr<[[ID1]]>) -> !llvm.ptr<struct<(i32, f32)>, 1> {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr<[[ACCESSORSTR]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr<[[ID1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.add %[[VAL_7]], %[[VAL_6]] : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0] : (!llvm.ptr<[[ACCESSORSTR]]>) -> !llvm.ptr<ptr<struct<(i32, f32)>, 1>>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<ptr<struct<(i32, f32)>, 1>>
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_10]][%[[VAL_8]]] : (!llvm.ptr<struct<(i32, f32)>, 1>, i64) -> !llvm.ptr<struct<(i32, f32)>, 1>
// CHECK-NEXT:      llvm.return %[[VAL_11]] : !llvm.ptr<struct<(i32, f32)>, 1>
// CHECK-NEXT:    }
func.func @test_struct(%acc: memref<?x!sycl_accessor_1_struct_rw_gb>, %idx: memref<?x!sycl_id_1_>) -> !llvm.ptr<struct<(i32, f32)>, 1> {
  %0 = sycl.accessor.subscript %acc[%idx] { ArgumentTypes = [memref<?x!sycl_accessor_1_struct_rw_gb>, memref<?x!sycl_id_1_>], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"id" }  : (memref<?x!sycl_accessor_1_struct_rw_gb>, memref<?x!sycl_id_1_>) -> !llvm.ptr<struct<(i32, f32)>, 1>
  return %0 : !llvm.ptr<struct<(i32, f32)>, 1>
}

// -----


//===----------------------------------------------------------------------===//
// sycl.accessor.subscript with id offset and atomic return type
//===----------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_ato_gb = !sycl.accessor<[1, i32, atomic, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_atomic_i32_1_ = !sycl.atomic<[i32, 1], (memref<?xi32, 1>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR1:.*]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: !llvm.ptr<[[ID1:.*]]>) -> !llvm.[[ATOM1:.*]] {
// CHECK-DAG:       %[[VAL_2:.*]] = llvm.mlir.undef : !llvm.[[ATOM1]]
// CHECK-DAG:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr<[[ACCESSOR1]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_5:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0, 0, 0] : (!llvm.ptr<[[ID1]]>) -> !llvm.ptr<i64>
// CHECK:           %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_8:.*]] = llvm.mul %[[VAL_3]], %[[VAL_5]]  : i64
// CHECK:           %[[VAL_9:.*]] = llvm.add %[[VAL_8]], %[[VAL_7]]  : i64
// CHECK:           %[[VAL_10:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0] : (!llvm.ptr<[[ACCESSOR1]]>) -> !llvm.ptr<ptr<i32, 1>>
// CHECK:           %[[VAL_11:.*]] = llvm.load %[[VAL_10]] : !llvm.ptr<ptr<i32, 1>>
// CHECK:           %[[VAL_12:.*]] = llvm.getelementptr inbounds %[[VAL_11]]{{\[}}%[[VAL_9]]] : (!llvm.ptr<i32, 1>, i64) -> !llvm.ptr<i32, 1>
// CHECK:           %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_12]], %[[VAL_2]][0] : !llvm.[[ATOM1]]
// CHECK:           llvm.return %[[VAL_13]] : !llvm.[[ATOM1]]
func.func @test(%acc: memref<?x!sycl_accessor_1_i32_ato_gb>, %idx: memref<?x!sycl_id_1_>) -> !sycl_atomic_i32_1_ {
  %0 = sycl.accessor.subscript %acc[%idx] { ArgumentTypes = [memref<?x!sycl_accessor_1_i32_ato_gb>, memref<?x!sycl_id_1_>], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"id" }  : (memref<?x!sycl_accessor_1_i32_ato_gb>, memref<?x!sycl_id_1_>) -> !sycl_atomic_i32_1_
  return %0 : !sycl_atomic_i32_1_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_nd_range_1_ = !sycl.nd_range<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[NDRANGE1:.*]]>) -> !llvm.[[RANGE1]] {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr<[[NDRANGE1]]>) -> !llvm.ptr<[[RANGE1]]>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<[[RANGE1]]>
// CHECK-NEXT:       llvm.return %[[VAL_2]] : !llvm.[[RANGE1]]
// CHECK-NEXT:     }
func.func @test(%nd: memref<?x!sycl_nd_range_1_>) -> !sycl_range_1_ {
  %0 = sycl.nd_range.get_global_range(%nd) { ArgumentTypes = [memref<?x!sycl_nd_range_1_>], FunctionName = @"get_global_range", MangledFunctionName = @"get_global_range", TypeName = @"nd_range" }  : (memref<?x!sycl_nd_range_1_>) -> !sycl_range_1_
  return %0 : !sycl_range_1_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_nd_range_1_ = !sycl.nd_range<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[NDRANGE1]]>) -> !llvm.[[RANGE1]] {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr<[[NDRANGE1]]>) -> !llvm.ptr<[[RANGE1]]>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<[[RANGE1]]>
// CHECK-NEXT:       llvm.return %[[VAL_2]] : !llvm.[[RANGE1]]
// CHECK-NEXT:     }
func.func @test(%nd: memref<?x!sycl_nd_range_1_>) -> !sycl_range_1_ {
  %0 = sycl.nd_range.get_local_range(%nd) { ArgumentTypes = [memref<?x!sycl_nd_range_1_>], FunctionName = @"get_local_range", MangledFunctionName = @"get_local_range", TypeName = @"nd_range" }  : (memref<?x!sycl_nd_range_1_>) -> !sycl_range_1_
  return %0 : !sycl_range_1_
}

// -----

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_nd_range_3_ = !sycl.nd_range<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[NDRANGE3:.*]]>) -> !llvm.[[RANGE3]] {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.[[RANGE3]] : (i64) -> !llvm.ptr<[[RANGE3]]>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0] : (!llvm.ptr<[[NDRANGE3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 0] : (!llvm.ptr<[[NDRANGE3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_7:.*]] = llvm.udiv %[[VAL_4]], %[[VAL_6]] : i64
// CHECK-NEXT:       %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0, 0, 0] : (!llvm.ptr<[[RANGE3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       llvm.store %[[VAL_7]], %[[VAL_8]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 1] : (!llvm.ptr<[[NDRANGE3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 1] : (!llvm.ptr<[[NDRANGE3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_13:.*]] = llvm.udiv %[[VAL_10]], %[[VAL_12]] : i64
// CHECK-NEXT:       %[[VAL_14:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0, 0, 1] : (!llvm.ptr<[[RANGE3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       llvm.store %[[VAL_13]], %[[VAL_14]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_15:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 2] : (!llvm.ptr<[[NDRANGE3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_16:.*]] = llvm.load %[[VAL_15]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_17:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 2] : (!llvm.ptr<[[NDRANGE3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_18:.*]] = llvm.load %[[VAL_17]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_19:.*]] = llvm.udiv %[[VAL_16]], %[[VAL_18]] : i64
// CHECK-NEXT:       %[[VAL_20:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0, 0, 2] : (!llvm.ptr<[[RANGE3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       llvm.store %[[VAL_19]], %[[VAL_20]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_21:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<[[RANGE3]]>
// CHECK-NEXT:       llvm.return %[[VAL_21]] : !llvm.[[RANGE3]]
// CHECK-NEXT:     }
func.func @test(%nd: memref<?x!sycl_nd_range_3_>) -> !sycl_range_3_ {
  %0 = sycl.nd_range.get_group_range(%nd) { ArgumentTypes = [memref<?x!sycl_nd_range_3_>], FunctionName = @"get_group_range", MangledFunctionName = @"get_group_range", TypeName = @"nd_range" }  : (memref<?x!sycl_nd_range_3_>) -> !sycl_range_3_
  return %0 : !sycl_range_3_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, false], (!sycl_item_base_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ITEM1:.*]]>) -> !llvm.[[ID1]] {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1] : (!llvm.ptr<[[ITEM1]]>) -> !llvm.ptr<[[ID1]]>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<[[ID1]]>
// CHECK-NEXT:       llvm.return %[[VAL_2]] : !llvm.[[ID1]]
// CHECK-NEXT:     }
func.func @test(%item: memref<?x!sycl_item_1_>) -> !sycl_id_1_ {
  %0 = sycl.item.get_id(%item) { ArgumentTypes = [memref<?x!sycl_item_1_>], FunctionName = @"get_id", MangledFunctionName = @"get_id", TypeName = @"item" }  : (memref<?x!sycl_item_1_>) -> !sycl_id_1_
  return %0 : !sycl_id_1_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, false], (!sycl_item_base_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ITEM1]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[ITEM1]]>, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:       llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:     }
func.func @test(%item: memref<?x!sycl_item_1_>, %i: i32) -> i64 {
  %0 = sycl.item.get_id(%item, %i) { ArgumentTypes = [memref<?x!sycl_item_1_>, i32], FunctionName = @"get_id", MangledFunctionName = @"get_id", TypeName = @"item" }  : (memref<?x!sycl_item_1_>, i32) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, false], (!sycl_item_base_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ITEM1]]>) -> i64 {
// CHECK:            %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:            %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[ITEM1]]>, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:       llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:     }
func.func @test(%item: memref<?x!sycl_item_1_>) -> i64 {
  %0 = sycl.item.get_id(%item) { ArgumentTypes = [memref<?x!sycl_item_1_>], FunctionName = @"operator unsigned long", MangledFunctionName = @"operator unsigned long", TypeName = @"item" }  : (memref<?x!sycl_item_1_>) -> i64
  return %0 : i64
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, false], (!sycl_item_base_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ITEM1]]>) -> !llvm.[[RANGE1]] {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0] : (!llvm.ptr<[[ITEM1]]>) -> !llvm.ptr<[[RANGE1]]>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<[[RANGE1]]>
// CHECK-NEXT:       llvm.return %[[VAL_2]] : !llvm.[[RANGE1]]
// CHECK-NEXT:     }
func.func @test(%item: memref<?x!sycl_item_1_>) -> !sycl_range_1_ {
  %0 = sycl.item.get_range(%item) { ArgumentTypes = [memref<?x!sycl_item_1_>], FunctionName = @"get_range", MangledFunctionName = @"get_range", TypeName = @"item" }  : (memref<?x!sycl_item_1_>) -> !sycl_range_1_
  return %0 : !sycl_range_1_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_item_base_1_ = !sycl.item_base<[1, false], (!sycl_range_1_, !sycl_id_1_)>
!sycl_item_1_ = !sycl.item<[1, false], (!sycl_item_base_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ITEM1]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[ITEM1]]>, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:       llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:     }
func.func @test(%item: memref<?x!sycl_item_1_>, %i: i32) -> i64 {
  %0 = sycl.item.get_range(%item, %i) { ArgumentTypes = [memref<?x!sycl_item_1_>, i32], FunctionName = @"get_range", MangledFunctionName = @"get_range", TypeName = @"item" }  : (memref<?x!sycl_item_1_>, i32) -> i64
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
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[ITEM1]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ITEM1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK-NEXT:       llvm.return %[[VAL_2]] : i64
// CHECK-NEXT:     }
func.func @test_1(%item: memref<?x!sycl_item_1_>) -> i64 {
  %0 = sycl.item.get_linear_id(%item) { ArgumentTypes = [memref<?x!sycl_item_1_>], FunctionName = @"get_linear_id", MangledFunctionName = @"get_linear_id", TypeName = @"item" }  : (memref<?x!sycl_item_1_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[ITEM2:.*]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ITEM2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 1] : (!llvm.ptr<[[ITEM2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:       %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 1] : (!llvm.ptr<[[ITEM2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_8:.*]] = llvm.add %[[VAL_5]], %[[VAL_7]] : i64
// CHECK-NEXT:       llvm.return %[[VAL_8]] : i64
// CHECK-NEXT:     }
func.func @test_2(%item: memref<?x!sycl_item_2_>) -> i64 {
  %0 = sycl.item.get_linear_id(%item) { ArgumentTypes = [memref<?x!sycl_item_2_>], FunctionName = @"get_linear_id", MangledFunctionName = @"get_linear_id", TypeName = @"item" }  : (memref<?x!sycl_item_2_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[ITEM3:.*]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 1] : (!llvm.ptr<[[ITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:       %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 2] : (!llvm.ptr<[[ITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_8:.*]] = llvm.mul %[[VAL_5]], %[[VAL_7]] : i64
// CHECK-NEXT:       %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 1] : (!llvm.ptr<[[ITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_11:.*]] = llvm.mul %[[VAL_10]], %[[VAL_7]] : i64
// CHECK-NEXT:       %[[VAL_12:.*]] = llvm.add %[[VAL_8]], %[[VAL_11]] : i64
// CHECK-NEXT:       %[[VAL_13:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 2] : (!llvm.ptr<[[ITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_15:.*]] = llvm.add %[[VAL_12]], %[[VAL_14]] : i64
// CHECK-NEXT:       llvm.return %[[VAL_15]] : i64
// CHECK-NEXT:     }
func.func @test_3(%item: memref<?x!sycl_item_3_>) -> i64 {
  %0 = sycl.item.get_linear_id(%item) { ArgumentTypes = [memref<?x!sycl_item_3_>], FunctionName = @"get_linear_id", MangledFunctionName = @"get_linear_id", TypeName = @"item" }  : (memref<?x!sycl_item_3_>) -> i64
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
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[ITEM1:.*]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ITEM1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr<[[ITEM1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_5:.*]] = llvm.sub %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:       llvm.return %[[VAL_5]] : i64
// CHECK-NEXT:     }
func.func @test_1(%item: memref<?x!sycl_item_1_>) -> i64 {
  %0 = sycl.item.get_linear_id(%item) { ArgumentTypes = [memref<?x!sycl_item_1_>], FunctionName = @"get_linear_id", MangledFunctionName = @"get_linear_id", TypeName = @"item" }  : (memref<?x!sycl_item_1_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[ITEM2:.*]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ITEM2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr<[[ITEM2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_5:.*]] = llvm.sub %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:       %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 1] : (!llvm.ptr<[[ITEM2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_8:.*]] = llvm.mul %[[VAL_5]], %[[VAL_7]] : i64
// CHECK-NEXT:       %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 1] : (!llvm.ptr<[[ITEM2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 1] : (!llvm.ptr<[[ITEM2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_13:.*]] = llvm.sub %[[VAL_10]], %[[VAL_12]] : i64
// CHECK-NEXT:       %[[VAL_14:.*]] = llvm.add %[[VAL_8]], %[[VAL_13]] : i64
// CHECK-NEXT:       llvm.return %[[VAL_14]] : i64
// CHECK-NEXT:     }
func.func @test_2(%item: memref<?x!sycl_item_2_>) -> i64 {
  %0 = sycl.item.get_linear_id(%item) { ArgumentTypes = [memref<?x!sycl_item_2_>], FunctionName = @"get_linear_id", MangledFunctionName = @"get_linear_id", TypeName = @"item" }  : (memref<?x!sycl_item_2_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[ITEM3:.*]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[ITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 0] : (!llvm.ptr<[[ITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_5:.*]] = llvm.sub %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:       %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 1] : (!llvm.ptr<[[ITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_8:.*]] = llvm.mul %[[VAL_5]], %[[VAL_7]] : i64
// CHECK-NEXT:       %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 2] : (!llvm.ptr<[[ITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_11:.*]] = llvm.mul %[[VAL_8]], %[[VAL_10]] : i64
// CHECK-NEXT:       %[[VAL_12:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 1] : (!llvm.ptr<[[ITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_13:.*]] = llvm.load %[[VAL_12]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_14:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 1] : (!llvm.ptr<[[ITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_15:.*]] = llvm.load %[[VAL_14]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_16:.*]] = llvm.sub %[[VAL_13]], %[[VAL_15]] : i64
// CHECK-NEXT:       %[[VAL_17:.*]] = llvm.mul %[[VAL_16]], %[[VAL_10]] : i64
// CHECK-NEXT:       %[[VAL_18:.*]] = llvm.add %[[VAL_11]], %[[VAL_17]] : i64
// CHECK-NEXT:       %[[VAL_19:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 1, 0, 0, 2] : (!llvm.ptr<[[ITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_20:.*]] = llvm.load %[[VAL_19]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_21:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 2, 0, 0, 2] : (!llvm.ptr<[[ITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_22:.*]] = llvm.load %[[VAL_21]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_23:.*]] = llvm.sub %[[VAL_20]], %[[VAL_22]] : i64
// CHECK-NEXT:       %[[VAL_24:.*]] = llvm.add %[[VAL_18]], %[[VAL_23]] : i64
// CHECK-NEXT:       llvm.return %[[VAL_24]] : i64
// CHECK-NEXT:     }
func.func @test_3(%item: memref<?x!sycl_item_3_>) -> i64 {
  %0 = sycl.item.get_linear_id(%item) { ArgumentTypes = [memref<?x!sycl_item_3_>], FunctionName = @"get_linear_id", MangledFunctionName = @"get_linear_id", TypeName = @"item" }  : (memref<?x!sycl_item_3_>) -> i64
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
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[NDITEM1:.*]]>) -> !llvm.[[ID1]] {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1] : (!llvm.ptr<[[NDITEM1]]>) -> !llvm.ptr<[[ID1]]>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<[[ID1]]>
// CHECK-NEXT:       llvm.return %[[VAL_2]] : !llvm.[[ID1]]
// CHECK-NEXT:     }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>) -> !sycl_id_1_ {
  %0 = sycl.nd_item.get_global_id(%nd) { ArgumentTypes = [memref<?x!sycl_nd_item_1_>], FunctionName = @"get_global_id", MangledFunctionName = @"get_global_id", TypeName = @"nd_item" }  : (memref<?x!sycl_nd_item_1_>) -> !sycl_id_1_
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
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[NDITEM1]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[NDITEM1]]>, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:       llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:     }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>, %i: i32) -> i64 {
  %0 = sycl.nd_item.get_global_id(%nd, %i) { ArgumentTypes = [memref<?x!sycl_nd_item_1_>, i32], FunctionName = @"get_global_id", MangledFunctionName = @"get_global_id", TypeName = @"nd_item" }  : (memref<?x!sycl_nd_item_1_>, i32) -> i64
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
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[NDITEM1]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[NDITEM1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK-NEXT:       llvm.return %[[VAL_2]] : i64
// CHECK-NEXT:     }
func.func @test_1(%nd: memref<?x!sycl_nd_item_1_>) -> i64 {
  %0 = sycl.nd_item.get_global_linear_id(%nd) { ArgumentTypes = [memref<?x!sycl_nd_item_1_>], FunctionName = @"get_global_linear_id", MangledFunctionName = @"get_global_linear_id", TypeName = @"nd_item" }  : (memref<?x!sycl_nd_item_1_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[NDITEM2:.*]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[NDITEM2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 0, 1] : (!llvm.ptr<[[NDITEM2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:       %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1, 0, 0, 1] : (!llvm.ptr<[[NDITEM2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_8:.*]] = llvm.add %[[VAL_5]], %[[VAL_7]] : i64
// CHECK-NEXT:       llvm.return %[[VAL_8]] : i64
// CHECK-NEXT:     }
func.func @test_2(%nd: memref<?x!sycl_nd_item_2_>) -> i64 {
  %0 = sycl.nd_item.get_global_linear_id(%nd) { ArgumentTypes = [memref<?x!sycl_nd_item_2_>], FunctionName = @"get_global_linear_id", MangledFunctionName = @"get_global_linear_id", TypeName = @"nd_item" }  : (memref<?x!sycl_nd_item_2_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[NDITEM3:.*]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1, 0, 0, 0] : (!llvm.ptr<[[NDITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 0, 1] : (!llvm.ptr<[[NDITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:       %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0, 0, 0, 2] : (!llvm.ptr<[[NDITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_8:.*]] = llvm.mul %[[VAL_5]], %[[VAL_7]] : i64
// CHECK-NEXT:       %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1, 0, 0, 1] : (!llvm.ptr<[[NDITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_11:.*]] = llvm.mul %[[VAL_10]], %[[VAL_7]] : i64
// CHECK-NEXT:       %[[VAL_12:.*]] = llvm.add %[[VAL_8]], %[[VAL_11]] : i64
// CHECK-NEXT:       %[[VAL_13:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 1, 0, 0, 2] : (!llvm.ptr<[[NDITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_15:.*]] = llvm.add %[[VAL_12]], %[[VAL_14]] : i64
// CHECK-NEXT:       llvm.return %[[VAL_15]] : i64
// CHECK-NEXT:     }
func.func @test_3(%nd: memref<?x!sycl_nd_item_3_>) -> i64 {
  %0 = sycl.nd_item.get_global_linear_id(%nd) { ArgumentTypes = [memref<?x!sycl_nd_item_3_>], FunctionName = @"get_global_linear_id", MangledFunctionName = @"get_global_linear_id", TypeName = @"nd_item" }  : (memref<?x!sycl_nd_item_3_>) -> i64
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
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[NDITEM1]]>) -> !llvm.[[ID1]] {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 1] : (!llvm.ptr<[[NDITEM1]]>) -> !llvm.ptr<[[ID1]]>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<[[ID1]]>
// CHECK-NEXT:       llvm.return %[[VAL_2]] : !llvm.[[ID1]]
// CHECK-NEXT:     }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>) -> !sycl_id_1_ {
  %0 = sycl.nd_item.get_local_id(%nd) { ArgumentTypes = [memref<?x!sycl_nd_item_1_>], FunctionName = @"get_local_id", MangledFunctionName = @"get_local_id", TypeName = @"nd_item" }  : (memref<?x!sycl_nd_item_1_>) -> !sycl_id_1_
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
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[NDITEM1]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 1, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[NDITEM1]]>, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:       llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:     }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>, %i: i32) -> i64 {
  %0 = sycl.nd_item.get_local_id(%nd, %i) { ArgumentTypes = [memref<?x!sycl_nd_item_1_>, i32], FunctionName = @"get_local_id", MangledFunctionName = @"get_local_id", TypeName = @"nd_item" }  : (memref<?x!sycl_nd_item_1_>, i32) -> i64
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
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[NDITEM1]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 1, 0, 0, 0] : (!llvm.ptr<[[NDITEM1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK-NEXT:       llvm.return %[[VAL_2]] : i64
// CHECK-NEXT:     }
func.func @test_1(%nd: memref<?x!sycl_nd_item_1_>) -> i64 {
  %0 = sycl.nd_item.get_local_linear_id(%nd) { ArgumentTypes = [memref<?x!sycl_nd_item_1_>], FunctionName = @"get_local_linear_id", MangledFunctionName = @"get_local_linear_id", TypeName = @"nd_item" }  : (memref<?x!sycl_nd_item_1_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[NDITEM2]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 1, 0, 0, 0] : (!llvm.ptr<[[NDITEM2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 0, 0, 1] : (!llvm.ptr<[[NDITEM2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:       %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 1, 0, 0, 1] : (!llvm.ptr<[[NDITEM2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_8:.*]] = llvm.add %[[VAL_5]], %[[VAL_7]] : i64
// CHECK-NEXT:       llvm.return %[[VAL_8]] : i64
// CHECK-NEXT:     }
func.func @test_2(%nd: memref<?x!sycl_nd_item_2_>) -> i64 {
  %0 = sycl.nd_item.get_local_linear_id(%nd) { ArgumentTypes = [memref<?x!sycl_nd_item_2_>], FunctionName = @"get_local_linear_id", MangledFunctionName = @"get_local_linear_id", TypeName = @"nd_item" }  : (memref<?x!sycl_nd_item_2_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[NDITEM3]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 1, 0, 0, 0] : (!llvm.ptr<[[NDITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 0, 0, 1] : (!llvm.ptr<[[NDITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:       %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 0, 0, 2] : (!llvm.ptr<[[NDITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_8:.*]] = llvm.mul %[[VAL_5]], %[[VAL_7]] : i64
// CHECK-NEXT:       %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 1, 0, 0, 1] : (!llvm.ptr<[[NDITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_11:.*]] = llvm.mul %[[VAL_10]], %[[VAL_7]] : i64
// CHECK-NEXT:       %[[VAL_12:.*]] = llvm.add %[[VAL_8]], %[[VAL_11]] : i64
// CHECK-NEXT:       %[[VAL_13:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 1, 0, 0, 2] : (!llvm.ptr<[[NDITEM3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_15:.*]] = llvm.add %[[VAL_12]], %[[VAL_14]] : i64
// CHECK-NEXT:       llvm.return %[[VAL_15]] : i64
// CHECK-NEXT:     }
func.func @test_3(%nd: memref<?x!sycl_nd_item_3_>) -> i64 {
  %0 = sycl.nd_item.get_local_linear_id(%nd) { ArgumentTypes = [memref<?x!sycl_nd_item_3_>], FunctionName = @"get_local_linear_id", MangledFunctionName = @"get_local_linear_id", TypeName = @"nd_item" }  : (memref<?x!sycl_nd_item_3_>) -> i64
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
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[NDITEM1]]>) -> !llvm.[[GROUP1:.*]] {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2] : (!llvm.ptr<[[NDITEM1]]>) -> !llvm.ptr<[[GROUP1]]>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<[[GROUP1]]>
// CHECK-NEXT:       llvm.return %[[VAL_2]] : !llvm.[[GROUP1]]
// CHECK-NEXT:     }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>) -> !sycl_group_1_ {
  %0 = sycl.nd_item.get_group(%nd) { ArgumentTypes = [memref<?x!sycl_nd_item_1_>], FunctionName = @"get_group", MangledFunctionName = @"get_group", TypeName = @"nd_item" }  : (memref<?x!sycl_nd_item_1_>) -> !sycl_group_1_
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
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[NDITEM1]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 3, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[NDITEM1]]>, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:       llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:     }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>, %i: i32) -> i64 {
  %0 = sycl.nd_item.get_group(%nd, %i) { ArgumentTypes = [memref<?x!sycl_nd_item_1_>, i32], FunctionName = @"get_group", MangledFunctionName = @"get_group", TypeName = @"nd_item" }  : (memref<?x!sycl_nd_item_1_>, i32) -> i64
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
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[NDITEM1]]>) -> !llvm.[[RANGE1]] {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 2] : (!llvm.ptr<[[NDITEM1]]>) -> !llvm.ptr<[[RANGE1]]>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<[[RANGE1]]>
// CHECK-NEXT:       llvm.return %[[VAL_2]] : !llvm.[[RANGE1]]
// CHECK-NEXT:     }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>) -> !sycl_range_1_ {
  %0 = sycl.nd_item.get_group_range(%nd) { ArgumentTypes = [memref<?x!sycl_nd_item_1_>], FunctionName = @"get_group_range", MangledFunctionName = @"get_group_range", TypeName = @"nd_item" }  : (memref<?x!sycl_nd_item_1_>) -> !sycl_range_1_
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
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[NDITEM1]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 2, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[NDITEM1]]>, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:       llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:     }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>, %i: i32) -> i64 {
  %0 = sycl.nd_item.get_group_range(%nd, %i) { ArgumentTypes = [memref<?x!sycl_nd_item_1_>, i32], FunctionName = @"get_group_range", MangledFunctionName = @"get_group_range", TypeName = @"nd_item" }  : (memref<?x!sycl_nd_item_1_>, i32) -> i64
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
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[NDITEM1]]>) -> !llvm.[[RANGE1]] {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0] : (!llvm.ptr<[[NDITEM1]]>) -> !llvm.ptr<[[RANGE1]]>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<[[RANGE1]]>
// CHECK-NEXT:       llvm.return %[[VAL_2]] : !llvm.[[RANGE1]]
// CHECK-NEXT:     }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>) -> !sycl_range_1_ {
  %0 = sycl.nd_item.get_local_range(%nd) { ArgumentTypes = [memref<?x!sycl_nd_item_1_>], FunctionName = @"get_local_range", MangledFunctionName = @"get_local_range", TypeName = @"nd_item" }  : (memref<?x!sycl_nd_item_1_>) -> !sycl_range_1_
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
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[NDITEM1]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[NDITEM1]]>, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:       llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:     }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>, %i: i32) -> i64 {
  %0 = sycl.nd_item.get_local_range(%nd, %i) { ArgumentTypes = [memref<?x!sycl_nd_item_1_>, i32], FunctionName = @"get_local_range", MangledFunctionName = @"get_local_range", TypeName = @"nd_item" }  : (memref<?x!sycl_nd_item_1_>, i32) -> i64
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
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[NDITEM1]]>) -> !llvm.[[NDRANGE1]] {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.[[NDRANGE1]] : (i32) -> !llvm.ptr<[[NDRANGE1]]>
// CHECK-DAG:        %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0] : (!llvm.ptr<[[NDRANGE1]]>) -> !llvm.ptr<[[RANGE1]]>
// CHECK-DAG:        %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 0] : (!llvm.ptr<[[NDITEM1]]>) -> !llvm.ptr<[[RANGE1]]>
// CHECK-DAG:        %[[VAL_5:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr<[[RANGE1]]>
// CHECK-NEXT:       llvm.store %[[VAL_5]], %[[VAL_3]] : !llvm.ptr<[[RANGE1]]>
// CHECK-DAG:        %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 1] : (!llvm.ptr<[[NDRANGE1]]>) -> !llvm.ptr<[[RANGE1]]>
// CHECK-DAG:        %[[VAL_7:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0] : (!llvm.ptr<[[NDITEM1]]>) -> !llvm.ptr<[[RANGE1]]>
// CHECK-DAG:        %[[VAL_8:.*]] = llvm.load %[[VAL_7]] : !llvm.ptr<[[RANGE1]]>
// CHECK-NEXT:       llvm.store %[[VAL_8]], %[[VAL_6]] : !llvm.ptr<[[RANGE1]]>
// CHECK-DAG:        %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 2] : (!llvm.ptr<[[NDRANGE1]]>) -> !llvm.ptr<[[ID1]]>
// CHECK-DAG:        %[[VAL_10:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0, 0, 2] : (!llvm.ptr<[[NDITEM1]]>) -> !llvm.ptr<[[ID1]]>
// CHECK-DAG:        %[[VAL_11:.*]] = llvm.load %[[VAL_10]] : !llvm.ptr<[[ID1]]>
// CHECK-NEXT:       llvm.store %[[VAL_11]], %[[VAL_9]] : !llvm.ptr<[[ID1]]>
// CHECK-NEXT:       %[[VAL_12:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<[[NDRANGE1]]>
// CHECK-NEXT:       llvm.return %[[VAL_12]] : !llvm.[[NDRANGE1]]
// CHECK-NEXT:     }
func.func @test(%nd: memref<?x!sycl_nd_item_1_>) -> !sycl_nd_range_1_ {
  %0 = sycl.nd_item.get_nd_range(%nd) { ArgumentTypes = [memref<?x!sycl_nd_item_1_>], FunctionName = @"get_nd_range", MangledFunctionName = @"get_nd_range", TypeName = @"nd_item" }  : (memref<?x!sycl_nd_item_1_>) -> !sycl_nd_range_1_
  return %0 : !sycl_nd_range_1_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[GROUP1]]>) -> !llvm.[[ID1]] {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 3] : (!llvm.ptr<[[GROUP1]]>) -> !llvm.ptr<[[ID1]]>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<[[ID1]]>
// CHECK-NEXT:       llvm.return %[[VAL_2]] : !llvm.[[ID1]]
// CHECK-NEXT:     }
func.func @test(%group: memref<?x!sycl_group_1_>) -> !sycl_id_1_ {
  %0 = sycl.group.get_group_id(%group) { ArgumentTypes = [memref<?x!sycl_group_1_>], FunctionName = @"get_group_id", MangledFunctionName = @"get_group_id", TypeName = @"group" }  : (memref<?x!sycl_group_1_>) -> !sycl_id_1_
  return %0 : !sycl_id_1_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[GROUP1]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 3, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[GROUP1]]>, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:       llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:     }
func.func @test(%group: memref<?x!sycl_group_1_>, %i: i32) -> i64 {
  %0 = sycl.group.get_group_id(%group, %i) { ArgumentTypes = [memref<?x!sycl_group_1_>, i32], FunctionName = @"get_group_id", MangledFunctionName = @"get_group_id", TypeName = @"group" }  : (memref<?x!sycl_group_1_>, i32) -> i64
  return %0 : i64
}

// -----

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_group_3_ = !sycl.group<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

module attributes {gpu.container} {
  gpu.module @kernels {
// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[GROUP3:.*]]>) -> !llvm.[[ID3:.*]] {
// CHECK:             %[[VAL_1:.*]] = llvm.mlir.addressof @__builtin_var_LocalInvocationId__ : !llvm.ptr<vector<3xi64>>
// CHECK:             %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<vector<3xi64>>
// CHECK:             %[[VAL_3:.*]] = llvm.mlir.null : !llvm.ptr<[[ID3]]>
// CHECK:             %[[VAL_4:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:             %[[VAL_5:.*]] = llvm.getelementptr %[[VAL_3]]{{\[}}%[[VAL_4]]] : (!llvm.ptr<[[ID3]]>, i64) -> !llvm.ptr<[[ID3]]>
// CHECK:             %[[VAL_6:.*]] = llvm.ptrtoint %[[VAL_5]] : !llvm.ptr<[[ID3]]> to i64
// CHECK:             %[[VAL_7:.*]] = llvm.alloca %[[VAL_6]] x !llvm.[[ID3]] : (i64) -> !llvm.ptr<[[ID3]]>
// CHECK-DAG:         %[[VAL_8:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG:         %[[VAL_9:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:         %[[VAL_10:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_11:.*]] = llvm.extractelement %[[VAL_2]]{{\[}}%[[VAL_10]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_12:.*]] = llvm.getelementptr inbounds %[[VAL_7]][0, 0, 0, %[[VAL_9]]] : (!llvm.ptr<[[ID3]]>, i32) -> !llvm.ptr<i64>
// CHECK:             %[[VAL_13:.*]] = llvm.getelementptr %[[VAL_12]]{{\[}}%[[VAL_8]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
// CHECK:             llvm.store %[[VAL_11]], %[[VAL_13]] : !llvm.ptr<i64>
// CHECK-DAG:         %[[VAL_14:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG:         %[[VAL_15:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:             %[[VAL_16:.*]] = llvm.extractelement %[[VAL_2]]{{\[}}%[[VAL_15]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_17:.*]] = llvm.getelementptr inbounds %[[VAL_7]][0, 0, 0, %[[VAL_14]]] : (!llvm.ptr<[[ID3]]>, i32) -> !llvm.ptr<i64>
// CHECK:             %[[VAL_18:.*]] = llvm.getelementptr %[[VAL_17]]{{\[}}%[[VAL_8]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
// CHECK:             llvm.store %[[VAL_16]], %[[VAL_18]] : !llvm.ptr<i64>
// CHECK-DAG:         %[[VAL_19:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-DAG:         %[[VAL_20:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:             %[[VAL_21:.*]] = llvm.extractelement %[[VAL_2]]{{\[}}%[[VAL_20]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_22:.*]] = llvm.getelementptr inbounds %[[VAL_7]][0, 0, 0, %[[VAL_19]]] : (!llvm.ptr<[[ID3]]>, i32) -> !llvm.ptr<i64>
// CHECK:             %[[VAL_23:.*]] = llvm.getelementptr %[[VAL_22]]{{\[}}%[[VAL_8]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
// CHECK:             llvm.store %[[VAL_21]], %[[VAL_23]] : !llvm.ptr<i64>
// CHECK:             %[[VAL_24:.*]] = llvm.getelementptr %[[VAL_7]]{{\[}}%[[VAL_8]]] : (!llvm.ptr<[[ID3]]>, i64) -> !llvm.ptr<[[ID3]]>
// CHECK:             %[[VAL_25:.*]] = llvm.load %[[VAL_24]] : !llvm.ptr<[[ID3]]>
// CHECK:             llvm.return %[[VAL_25]] : !llvm.[[ID3]]
// CHECK-NEXT:     }
    func.func @test(%group: memref<?x!sycl_group_3_>) -> !sycl_id_3_ {
      %0 = sycl.group.get_local_id(%group) { ArgumentTypes = [memref<?x!sycl_group_3_>], FunctionName = @"get_local_id", MangledFunctionName = @"get_local_id", TypeName = @"group" }  : (memref<?x!sycl_group_3_>) -> !sycl_id_3_
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
// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[GROUP1]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK:             %[[VAL_2:.*]] = llvm.mlir.addressof @__builtin_var_LocalInvocationId__ : !llvm.ptr<vector<3xi64>>
// CHECK:             %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<vector<3xi64>>
// CHECK-DAG:         %[[VAL_4:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
// CHECK-DAG:         %[[VAL_5:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_6:.*]] = llvm.extractelement %[[VAL_3]]{{\[}}%[[VAL_5]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_7:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:             %[[VAL_8:.*]] = llvm.insertelement %[[VAL_6]], %[[VAL_4]]{{\[}}%[[VAL_7]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_9:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:             %[[VAL_10:.*]] = llvm.extractelement %[[VAL_3]]{{\[}}%[[VAL_9]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_11:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:             %[[VAL_12:.*]] = llvm.insertelement %[[VAL_10]], %[[VAL_8]]{{\[}}%[[VAL_11]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_13:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:             %[[VAL_14:.*]] = llvm.extractelement %[[VAL_3]]{{\[}}%[[VAL_13]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_15:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:             %[[VAL_16:.*]] = llvm.insertelement %[[VAL_14]], %[[VAL_12]]{{\[}}%[[VAL_15]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_17:.*]] = llvm.extractelement %[[VAL_16]]{{\[}}%[[VAL_1]] : i32] : vector<3xi64>
// CHECK:             llvm.return %[[VAL_17]] : i64
// CHECK-NEXT:     }
    func.func @test(%group: memref<?x!sycl_group_1_>, %i: i32) -> i64 {
      %0 = sycl.group.get_local_id(%group, %i) { ArgumentTypes = [memref<?x!sycl_group_1_>, i32], FunctionName = @"get_local_id", MangledFunctionName = @"get_local_id", TypeName = @"group" }  : (memref<?x!sycl_group_1_>, i32) -> i64
      return %0 : i64
    }
  }
}

// -----

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_group_3_ = !sycl.group<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[GROUP3]]>) -> !llvm.[[RANGE3]] {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr<[[GROUP3]]>) -> !llvm.ptr<[[RANGE3]]>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<[[RANGE3]]>
// CHECK-NEXT:       llvm.return %[[VAL_2]] : !llvm.[[RANGE3]]
// CHECK-NEXT:     }
func.func @test(%group: memref<?x!sycl_group_3_>) -> !sycl_range_3_ {
  %0 = sycl.group.get_local_range(%group) { ArgumentTypes = [memref<?x!sycl_group_3_>], FunctionName = @"get_local_range", MangledFunctionName = @"get_local_range", TypeName = @"group" }  : (memref<?x!sycl_group_3_>) -> !sycl_range_3_
  return %0 : !sycl_range_3_
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[GROUP1]]>,
// CHECK-SAME:                    %[[VAL_1:.*]]: i32) -> i64 {
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, %[[VAL_1]]] : (!llvm.ptr<[[GROUP1]]>, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:       llvm.return %[[VAL_3]] : i64
// CHECK-NEXT:     }
func.func @test(%group: memref<?x!sycl_group_1_>, %i: i32) -> i64 {
  %0 = sycl.group.get_local_range(%group, %i) { ArgumentTypes = [memref<?x!sycl_group_1_>, i32], FunctionName = @"get_local_range", MangledFunctionName = @"get_local_range", TypeName = @"group" }  : (memref<?x!sycl_group_1_>, i32) -> i64
  return %0 : i64
}

// -----

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_group_3_ = !sycl.group<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[GROUP3]]>) -> !llvm.[[RANGE3]] {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr<[[GROUP3]]>) -> !llvm.ptr<[[RANGE3]]>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<[[RANGE3]]>
// CHECK-NEXT:       llvm.return %[[VAL_2]] : !llvm.[[RANGE3]]
// CHECK-NEXT:     }
func.func @test(%group: memref<?x!sycl_group_3_>) -> !sycl_range_3_ {
  %0 = sycl.group.get_max_local_range(%group) { ArgumentTypes = [memref<?x!sycl_group_3_>], FunctionName = @"get_max_local_range", MangledFunctionName = @"get_max_local_range", TypeName = @"group" }  : (memref<?x!sycl_group_3_>) -> !sycl_range_3_
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
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[GROUP1]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 3, 0, 0, 0] : (!llvm.ptr<[[GROUP1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK-NEXT:       llvm.return %[[VAL_2]] : i64
// CHECK-NEXT:     }
func.func @test_1(%group: memref<?x!sycl_group_1_>) -> i64 {
  %0 = sycl.group.get_group_linear_id(%group) { ArgumentTypes = [memref<?x!sycl_group_1_>], FunctionName = @"get_group_linear_id", MangledFunctionName = @"get_group_linear_id", TypeName = @"group" }  : (memref<?x!sycl_group_1_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[GROUP2:.*]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 3, 0, 0, 0] : (!llvm.ptr<[[GROUP2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 0, 0, 1] : (!llvm.ptr<[[GROUP2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:       %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 3, 0, 0, 1] : (!llvm.ptr<[[GROUP2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_8:.*]] = llvm.add %[[VAL_5]], %[[VAL_7]] : i64
// CHECK-NEXT:       llvm.return %[[VAL_8]] : i64
// CHECK-NEXT:     }
func.func @test_2(%group: memref<?x!sycl_group_2_>) -> i64 {
  %0 = sycl.group.get_group_linear_id(%group) { ArgumentTypes = [memref<?x!sycl_group_2_>], FunctionName = @"get_group_linear_id", MangledFunctionName = @"get_group_linear_id", TypeName = @"group" }  : (memref<?x!sycl_group_2_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[GROUP3]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 3, 0, 0, 0] : (!llvm.ptr<[[GROUP3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 0, 0, 1] : (!llvm.ptr<[[GROUP3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_5:.*]] = llvm.mul %[[VAL_2]], %[[VAL_4]] : i64
// CHECK-NEXT:       %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 0, 0, 2] : (!llvm.ptr<[[GROUP3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_7:.*]] = llvm.load %[[VAL_6]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_8:.*]] = llvm.mul %[[VAL_5]], %[[VAL_7]] : i64
// CHECK-NEXT:       %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 3, 0, 0, 1] : (!llvm.ptr<[[GROUP3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_11:.*]] = llvm.mul %[[VAL_10]], %[[VAL_7]] : i64
// CHECK-NEXT:       %[[VAL_12:.*]] = llvm.add %[[VAL_8]], %[[VAL_11]] : i64
// CHECK-NEXT:       %[[VAL_13:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 3, 0, 0, 2] : (!llvm.ptr<[[GROUP3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_15:.*]] = llvm.add %[[VAL_12]], %[[VAL_14]] : i64
// CHECK-NEXT:       llvm.return %[[VAL_15]] : i64
// CHECK-NEXT:     }
func.func @test_3(%group: memref<?x!sycl_group_3_>) -> i64 {
  %0 = sycl.group.get_group_linear_id(%group) { ArgumentTypes = [memref<?x!sycl_group_3_>], FunctionName = @"get_group_linear_id", MangledFunctionName = @"get_group_linear_id", TypeName = @"group" }  : (memref<?x!sycl_group_3_>) -> i64
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
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[GROUP1]]>) -> i64 {
// CHECK:             %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_2:.*]] = llvm.mlir.addressof @__builtin_var_LocalInvocationId__ : !llvm.ptr<vector<3xi64>>
// CHECK:             %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<vector<3xi64>>
// CHECK-DAG:         %[[VAL_4:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
// CHECK-DAG:         %[[VAL_5:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_6:.*]] = llvm.extractelement %[[VAL_3]]{{\[}}%[[VAL_5]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_7:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:             %[[VAL_8:.*]] = llvm.insertelement %[[VAL_6]], %[[VAL_4]]{{\[}}%[[VAL_7]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_9:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:             %[[VAL_10:.*]] = llvm.extractelement %[[VAL_3]]{{\[}}%[[VAL_9]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_11:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:             %[[VAL_12:.*]] = llvm.insertelement %[[VAL_10]], %[[VAL_8]]{{\[}}%[[VAL_11]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_13:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:             %[[VAL_14:.*]] = llvm.extractelement %[[VAL_3]]{{\[}}%[[VAL_13]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_15:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:             %[[VAL_16:.*]] = llvm.insertelement %[[VAL_14]], %[[VAL_12]]{{\[}}%[[VAL_15]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_17:.*]] = llvm.extractelement %[[VAL_16]]{{\[}}%[[VAL_1]] : i32] : vector<3xi64>
// CHECK:             llvm.return %[[VAL_17]] : i64
// CHECK-NEXT:     }
    func.func @test_1(%group: memref<?x!sycl_group_1_>) -> i64 {
      %0 = sycl.group.get_local_linear_id(%group) { ArgumentTypes = [memref<?x!sycl_group_1_>], FunctionName = @"get_local_linear_id", MangledFunctionName = @"get_local_linear_id", TypeName = @"group" }  : (memref<?x!sycl_group_1_>) -> i64
      return %0 : i64
    }

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_18:.*]]: !llvm.ptr<[[GROUP2]]>) -> i64 {
// CHECK:             %[[VAL_19:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_20:.*]] = llvm.mlir.addressof @__builtin_var_LocalInvocationId__ : !llvm.ptr<vector<3xi64>>
// CHECK:             %[[VAL_21:.*]] = llvm.load %[[VAL_20]] : !llvm.ptr<vector<3xi64>>
// CHECK-DAG:         %[[VAL_22:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
// CHECK-DAG:         %[[VAL_23:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_24:.*]] = llvm.extractelement %[[VAL_21]]{{\[}}%[[VAL_23]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_25:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:             %[[VAL_26:.*]] = llvm.insertelement %[[VAL_24]], %[[VAL_22]]{{\[}}%[[VAL_25]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_27:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:             %[[VAL_28:.*]] = llvm.extractelement %[[VAL_21]]{{\[}}%[[VAL_27]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_29:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:             %[[VAL_30:.*]] = llvm.insertelement %[[VAL_28]], %[[VAL_26]]{{\[}}%[[VAL_29]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_31:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:             %[[VAL_32:.*]] = llvm.extractelement %[[VAL_21]]{{\[}}%[[VAL_31]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_33:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:             %[[VAL_34:.*]] = llvm.insertelement %[[VAL_32]], %[[VAL_30]]{{\[}}%[[VAL_33]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_35:.*]] = llvm.extractelement %[[VAL_34]]{{\[}}%[[VAL_19]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_36:.*]] = llvm.getelementptr inbounds %[[VAL_18]][0, 2, 0, 0, 1] : (!llvm.ptr<[[GROUP2]]>) -> !llvm.ptr<i64>
// CHECK:             %[[VAL_37:.*]] = llvm.load %[[VAL_36]] : !llvm.ptr<i64>
// CHECK:             %[[VAL_38:.*]] = llvm.mul %[[VAL_35]], %[[VAL_37]]  : i64
// CHECK:             %[[VAL_39:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:             %[[VAL_40:.*]] = llvm.mlir.addressof @__builtin_var_LocalInvocationId__ : !llvm.ptr<vector<3xi64>>
// CHECK:             %[[VAL_41:.*]] = llvm.load %[[VAL_40]] : !llvm.ptr<vector<3xi64>>
// CHECK-DAG:         %[[VAL_42:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
// CHECK-DAG:         %[[VAL_43:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_44:.*]] = llvm.extractelement %[[VAL_41]]{{\[}}%[[VAL_43]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_45:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:             %[[VAL_46:.*]] = llvm.insertelement %[[VAL_44]], %[[VAL_42]]{{\[}}%[[VAL_45]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_47:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:             %[[VAL_48:.*]] = llvm.extractelement %[[VAL_41]]{{\[}}%[[VAL_47]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_49:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:             %[[VAL_50:.*]] = llvm.insertelement %[[VAL_48]], %[[VAL_46]]{{\[}}%[[VAL_49]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_51:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:             %[[VAL_52:.*]] = llvm.extractelement %[[VAL_41]]{{\[}}%[[VAL_51]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_53:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:             %[[VAL_54:.*]] = llvm.insertelement %[[VAL_52]], %[[VAL_50]]{{\[}}%[[VAL_53]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_55:.*]] = llvm.extractelement %[[VAL_54]]{{\[}}%[[VAL_39]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_56:.*]] = llvm.add %[[VAL_38]], %[[VAL_55]]  : i64
// CHECK:             llvm.return %[[VAL_56]] : i64
// CHECK-NEXT:     }
    func.func @test_2(%group: memref<?x!sycl_group_2_>) -> i64 {
      %0 = sycl.group.get_local_linear_id(%group) { ArgumentTypes = [memref<?x!sycl_group_2_>], FunctionName = @"get_local_linear_id", MangledFunctionName = @"get_local_linear_id", TypeName = @"group" }  : (memref<?x!sycl_group_2_>) -> i64
      return %0 : i64
    }

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_57:.*]]: !llvm.ptr<[[GROUP3]]>) -> i64 {
// CHECK:             %[[VAL_58:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_59:.*]] = llvm.mlir.addressof @__builtin_var_LocalInvocationId__ : !llvm.ptr<vector<3xi64>>
// CHECK:             %[[VAL_60:.*]] = llvm.load %[[VAL_59]] : !llvm.ptr<vector<3xi64>>
// CHECK-DAG:         %[[VAL_61:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
// CHECK-DAG:         %[[VAL_62:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_63:.*]] = llvm.extractelement %[[VAL_60]]{{\[}}%[[VAL_62]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_64:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:             %[[VAL_65:.*]] = llvm.insertelement %[[VAL_63]], %[[VAL_61]]{{\[}}%[[VAL_64]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_66:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:             %[[VAL_67:.*]] = llvm.extractelement %[[VAL_60]]{{\[}}%[[VAL_66]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_68:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:             %[[VAL_69:.*]] = llvm.insertelement %[[VAL_67]], %[[VAL_65]]{{\[}}%[[VAL_68]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_70:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:             %[[VAL_71:.*]] = llvm.extractelement %[[VAL_60]]{{\[}}%[[VAL_70]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_72:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:             %[[VAL_73:.*]] = llvm.insertelement %[[VAL_71]], %[[VAL_69]]{{\[}}%[[VAL_72]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_74:.*]] = llvm.extractelement %[[VAL_73]]{{\[}}%[[VAL_58]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_75:.*]] = llvm.getelementptr inbounds %[[VAL_57]][0, 2, 0, 0, 1] : (!llvm.ptr<[[GROUP3]]>) -> !llvm.ptr<i64>
// CHECK:             %[[VAL_76:.*]] = llvm.load %[[VAL_75]] : !llvm.ptr<i64>
// CHECK:             %[[VAL_77:.*]] = llvm.mul %[[VAL_74]], %[[VAL_76]]  : i64
// CHECK:             %[[VAL_78:.*]] = llvm.getelementptr inbounds %[[VAL_57]][0, 2, 0, 0, 2] : (!llvm.ptr<[[GROUP3]]>) -> !llvm.ptr<i64>
// CHECK:             %[[VAL_79:.*]] = llvm.load %[[VAL_78]] : !llvm.ptr<i64>
// CHECK:             %[[VAL_80:.*]] = llvm.mul %[[VAL_77]], %[[VAL_79]]  : i64
// CHECK:             %[[VAL_81:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:             %[[VAL_82:.*]] = llvm.mlir.addressof @__builtin_var_LocalInvocationId__ : !llvm.ptr<vector<3xi64>>
// CHECK:             %[[VAL_83:.*]] = llvm.load %[[VAL_82]] : !llvm.ptr<vector<3xi64>>
// CHECK-DAG:         %[[VAL_84:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
// CHECK-DAG:         %[[VAL_85:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_86:.*]] = llvm.extractelement %[[VAL_83]]{{\[}}%[[VAL_85]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_87:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:             %[[VAL_88:.*]] = llvm.insertelement %[[VAL_86]], %[[VAL_84]]{{\[}}%[[VAL_87]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_89:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:             %[[VAL_90:.*]] = llvm.extractelement %[[VAL_83]]{{\[}}%[[VAL_89]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_91:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:             %[[VAL_92:.*]] = llvm.insertelement %[[VAL_90]], %[[VAL_88]]{{\[}}%[[VAL_91]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_93:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:             %[[VAL_94:.*]] = llvm.extractelement %[[VAL_83]]{{\[}}%[[VAL_93]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_95:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:             %[[VAL_96:.*]] = llvm.insertelement %[[VAL_94]], %[[VAL_92]]{{\[}}%[[VAL_95]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_97:.*]] = llvm.extractelement %[[VAL_96]]{{\[}}%[[VAL_81]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_98:.*]] = llvm.mul %[[VAL_97]], %[[VAL_79]]  : i64
// CHECK:             %[[VAL_99:.*]] = llvm.add %[[VAL_80]], %[[VAL_98]]  : i64
// CHECK:             %[[VAL_100:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:             %[[VAL_101:.*]] = llvm.mlir.addressof @__builtin_var_LocalInvocationId__ : !llvm.ptr<vector<3xi64>>
// CHECK:             %[[VAL_102:.*]] = llvm.load %[[VAL_101]] : !llvm.ptr<vector<3xi64>>
// CHECK-DAG:         %[[VAL_103:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
// CHECK-DAG:         %[[VAL_104:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_105:.*]] = llvm.extractelement %[[VAL_102]]{{\[}}%[[VAL_104]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_106:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:             %[[VAL_107:.*]] = llvm.insertelement %[[VAL_105]], %[[VAL_103]]{{\[}}%[[VAL_106]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_108:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:             %[[VAL_109:.*]] = llvm.extractelement %[[VAL_102]]{{\[}}%[[VAL_108]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_110:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:             %[[VAL_111:.*]] = llvm.insertelement %[[VAL_109]], %[[VAL_107]]{{\[}}%[[VAL_110]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_112:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:             %[[VAL_113:.*]] = llvm.extractelement %[[VAL_102]]{{\[}}%[[VAL_112]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_114:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:             %[[VAL_115:.*]] = llvm.insertelement %[[VAL_113]], %[[VAL_111]]{{\[}}%[[VAL_114]] : i64] : vector<3xi64>
// CHECK:             %[[VAL_116:.*]] = llvm.extractelement %[[VAL_115]]{{\[}}%[[VAL_100]] : i32] : vector<3xi64>
// CHECK:             %[[VAL_117:.*]] = llvm.add %[[VAL_99]], %[[VAL_116]]  : i64
// CHECK:             llvm.return %[[VAL_117]] : i64
// CHECK-NEXT:     }
    func.func @test_3(%group: memref<?x!sycl_group_3_>) -> i64 {
      %0 = sycl.group.get_local_linear_id(%group) { ArgumentTypes = [memref<?x!sycl_group_3_>], FunctionName = @"get_local_linear_id", MangledFunctionName = @"get_local_linear_id", TypeName = @"group" }  : (memref<?x!sycl_group_3_>) -> i64
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
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[GROUP1]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 0, 0, 0] : (!llvm.ptr<[[GROUP1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_4:.*]] = llvm.mul %[[VAL_1]], %[[VAL_3]] : i64
// CHECK-NEXT:       llvm.return %[[VAL_4]] : i64
// CHECK-NEXT:     }
func.func @test_1(%group: memref<?x!sycl_group_1_>) -> i64 {
  %0 = sycl.group.get_group_linear_range(%group) { ArgumentTypes = [memref<?x!sycl_group_1_>], FunctionName = @"get_group_linear_range", MangledFunctionName = @"get_group_linear_range", TypeName = @"group" }  : (memref<?x!sycl_group_1_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[GROUP2]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 0, 0, 0] : (!llvm.ptr<[[GROUP2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_4:.*]] = llvm.mul %[[VAL_1]], %[[VAL_3]] : i64
// CHECK-NEXT:       %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 0, 0, 1] : (!llvm.ptr<[[GROUP2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_7:.*]] = llvm.mul %[[VAL_4]], %[[VAL_6]] : i64
// CHECK-NEXT:       llvm.return %[[VAL_7]] : i64
// CHECK-NEXT:     }
func.func @test_2(%group: memref<?x!sycl_group_2_>) -> i64 {
  %0 = sycl.group.get_group_linear_range(%group) { ArgumentTypes = [memref<?x!sycl_group_2_>], FunctionName = @"get_group_linear_range", MangledFunctionName = @"get_group_linear_range", TypeName = @"group" }  : (memref<?x!sycl_group_2_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[GROUP3]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 0, 0, 0] : (!llvm.ptr<[[GROUP3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_4:.*]] = llvm.mul %[[VAL_1]], %[[VAL_3]] : i64
// CHECK-NEXT:       %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 0, 0, 1] : (!llvm.ptr<[[GROUP3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_7:.*]] = llvm.mul %[[VAL_4]], %[[VAL_6]] : i64
// CHECK-NEXT:       %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2, 0, 0, 2] : (!llvm.ptr<[[GROUP3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_10:.*]] = llvm.mul %[[VAL_7]], %[[VAL_9]] : i64
// CHECK-NEXT:       llvm.return %[[VAL_10]] : i64
// CHECK-NEXT:     }
func.func @test_3(%group: memref<?x!sycl_group_3_>) -> i64 {
  %0 = sycl.group.get_group_linear_range(%group) { ArgumentTypes = [memref<?x!sycl_group_3_>], FunctionName = @"get_group_linear_range", MangledFunctionName = @"get_group_linear_range", TypeName = @"group" }  : (memref<?x!sycl_group_3_>) -> i64
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
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[GROUP1]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 0] : (!llvm.ptr<[[GROUP1]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_4:.*]] = llvm.mul %[[VAL_1]], %[[VAL_3]] : i64
// CHECK-NEXT:       llvm.return %[[VAL_4]] : i64
// CHECK-NEXT:     }
func.func @test_1(%group: memref<?x!sycl_group_1_>) -> i64 {
  %0 = sycl.group.get_local_linear_range(%group) { ArgumentTypes = [memref<?x!sycl_group_1_>], FunctionName = @"get_local_linear_range", MangledFunctionName = @"get_local_linear_range", TypeName = @"group" }  : (memref<?x!sycl_group_1_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[GROUP2]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 0] : (!llvm.ptr<[[GROUP2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_4:.*]] = llvm.mul %[[VAL_1]], %[[VAL_3]] : i64
// CHECK-NEXT:       %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 1] : (!llvm.ptr<[[GROUP2]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_7:.*]] = llvm.mul %[[VAL_4]], %[[VAL_6]] : i64
// CHECK-NEXT:       llvm.return %[[VAL_7]] : i64
// CHECK-NEXT:     }
func.func @test_2(%group: memref<?x!sycl_group_2_>) -> i64 {
  %0 = sycl.group.get_local_linear_range(%group) { ArgumentTypes = [memref<?x!sycl_group_2_>], FunctionName = @"get_local_linear_range", MangledFunctionName = @"get_local_linear_range", TypeName = @"group" }  : (memref<?x!sycl_group_2_>) -> i64
  return %0 : i64
}

// CHECK-LABEL:   llvm.func @test_3(
// CHECK-SAME:                      %[[VAL_0:.*]]: !llvm.ptr<[[GROUP3]]>) -> i64 {
// CHECK-NEXT:       %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:       %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 0] : (!llvm.ptr<[[GROUP3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_4:.*]] = llvm.mul %[[VAL_1]], %[[VAL_3]] : i64
// CHECK-NEXT:       %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 1] : (!llvm.ptr<[[GROUP3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_7:.*]] = llvm.mul %[[VAL_4]], %[[VAL_6]] : i64
// CHECK-NEXT:       %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1, 0, 0, 2] : (!llvm.ptr<[[GROUP3]]>) -> !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr<i64>
// CHECK-NEXT:       %[[VAL_10:.*]] = llvm.mul %[[VAL_7]], %[[VAL_9]] : i64
// CHECK-NEXT:       llvm.return %[[VAL_10]] : i64
// CHECK-NEXT:     }
func.func @test_3(%group: memref<?x!sycl_group_3_>) -> i64 {
  %0 = sycl.group.get_local_linear_range(%group) { ArgumentTypes = [memref<?x!sycl_group_3_>], FunctionName = @"get_local_linear_range", MangledFunctionName = @"get_local_linear_range", TypeName = @"group" }  : (memref<?x!sycl_group_3_>) -> i64
  return %0 : i64
}
