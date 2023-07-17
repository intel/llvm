// RUN: sycl-mlir-opt %s -canonicalize --split-input-file | FileCheck %s

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>

// CHECK-LABEL:   func.func @id_constructor_use_default() -> memref<?x!sycl_id_3_> {
// CHECK-NEXT:      %[[VAL_0:.*]] = sycl.id.constructor() : () -> memref<?x!sycl_id_3_>
// CHECK-NEXT:      return %[[VAL_0]] : memref<?x!sycl_id_3_>
// CHECK-NEXT:    }
func.func @id_constructor_use_default() -> memref<?x!sycl_id_3_> {
  %c0 = arith.constant 0 : index
  %id = sycl.id.constructor(%c0, %c0, %c0)
      : (index, index, index) -> memref<?x!sycl_id_3_>
  func.return %id : memref<?x!sycl_id_3_>
}

// -----

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_nd_range_3_ = !sycl.nd_range<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

// CHECK-LABEL:   func.func @nd_constructor_drop_zero_offset(
// CHECK-SAME:                                               %[[VAL_0:.*]]: memref<?x!sycl_range_3_>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: memref<?x!sycl_range_3_>) -> (memref<?x!sycl_nd_range_3_>, memref<?x!sycl_id_3_>, i64) {
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[VAL_3:.*]] = sycl.id.constructor() : () -> memref<?x!sycl_id_3_>
// CHECK-NEXT:      %[[VAL_4:.*]] = sycl.id.constructor(%[[VAL_3]]) : (memref<?x!sycl_id_3_>) -> memref<?x!sycl_id_3_>
// CHECK-NEXT:      sycl.constructor @id(%[[VAL_4]], %[[VAL_3]]) {MangledFunctionName = @foo} : (memref<?x!sycl_id_3_>, memref<?x!sycl_id_3_>)
// CHECK-NEXT:      %[[VAL_5:.*]] = sycl.id.get %[[VAL_3]]{{\[}}%[[VAL_2]]] : (memref<?x!sycl_id_3_>, i32) -> i64
// CHECK-NEXT:      %[[VAL_6:.*]] = sycl.nd_range.constructor(%[[VAL_0]], %[[VAL_1]]) : (memref<?x!sycl_range_3_>, memref<?x!sycl_range_3_>) -> memref<?x!sycl_nd_range_3_>
// CHECK-NEXT:      return %[[VAL_6]], %[[VAL_4]], %[[VAL_5]] : memref<?x!sycl_nd_range_3_>, memref<?x!sycl_id_3_>, i64
// CHECK-NEXT:    }
func.func @nd_constructor_drop_zero_offset(
    %globalSize: memref<?x!sycl_range_3_>, %localSize: memref<?x!sycl_range_3_>)
    -> (memref<?x!sycl_nd_range_3_>, memref<?x!sycl_id_3_>, i64) {
  %c0 = arith.constant 0 : i32
  %offset = sycl.id.constructor() : () -> memref<?x!sycl_id_3_>
  %cpy = sycl.id.constructor(%offset) : (memref<?x!sycl_id_3_>) -> memref<?x!sycl_id_3_>
  sycl.constructor @id(%cpy, %offset) {MangledFunctionName=@foo} : (memref<?x!sycl_id_3_>, memref<?x!sycl_id_3_>)
  %zero = sycl.id.get %offset[%c0] : (memref<?x!sycl_id_3_>, i32) -> i64
  %nd = sycl.nd_range.constructor(%globalSize, %localSize, %offset)
    : (memref<?x!sycl_range_3_>, memref<?x!sycl_range_3_>, memref<?x!sycl_id_3_>)
    -> memref<?x!sycl_nd_range_3_>
  func.return %nd, %cpy, %zero : memref<?x!sycl_nd_range_3_>, memref<?x!sycl_id_3_>, i64
}
