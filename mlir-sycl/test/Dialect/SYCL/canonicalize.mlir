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

// -----

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>

// CHECK-LABEL:   func.func @id_constructor_abstract_cast(
// CHECK-SAME:                                            %[[VAL_0:.*]]: memref<?x!sycl_range_3_>) -> memref<?x!sycl_id_3_> {
// CHECK:           %[[VAL_1:.*]] = sycl.id.constructor(%[[VAL_0]]) : (memref<?x!sycl_range_3_>) -> memref<?x!sycl_id_3_>
// CHECK:           return %[[VAL_1]] : memref<?x!sycl_id_3_>
// CHECK:         }

func.func @id_constructor_abstract_cast(%range: memref<?x!sycl_range_3_>) -> memref<?x!sycl_id_3_> {
  %cast = memref.memory_space_cast %range : memref<?x!sycl_range_3_> to memref<?x!sycl_range_3_, 4>
  %res = sycl.id.constructor(%cast) : (memref<?x!sycl_range_3_, 4>) -> memref<?x!sycl_id_3_>
  func.return %res : memref<?x!sycl_id_3_>
}

// -----

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>

// CHECK-LABEL:   func.func @range_constructor_abstract_cast(
// CHECK-SAME:                                               %[[VAL_0:.*]]: memref<?x!sycl_range_3_, #sycl.access.address_space<global>>) -> memref<?x!sycl_range_3_> {
// CHECK:           %[[VAL_1:.*]] = sycl.range.constructor(%[[VAL_0]]) : (memref<?x!sycl_range_3_, #sycl.access.address_space<global>>) -> memref<?x!sycl_range_3_>
// CHECK:           return %[[VAL_1]] : memref<?x!sycl_range_3_>
// CHECK:         }

func.func @range_constructor_abstract_cast(%range: memref<?x!sycl_range_3_, #sycl.access.address_space<global>>) -> memref<?x!sycl_range_3_> {
  %cast = memref.memory_space_cast %range : memref<?x!sycl_range_3_, #sycl.access.address_space<global>> to memref<?x!sycl_range_3_, #sycl.access.address_space<generic>>
  %res = sycl.range.constructor(%cast) : (memref<?x!sycl_range_3_, #sycl.access.address_space<generic>>) -> memref<?x!sycl_range_3_>
  func.return %res : memref<?x!sycl_range_3_>
}

// -----

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>
!sycl_nd_range_3_ = !sycl.nd_range<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

// CHECK-LABEL:   func.func @nd_range_constructor_abstract_cast(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: memref<?x!sycl_range_3_, #sycl.access.address_space<global>>, %[[VAL_1:.*]]: memref<?x!sycl_range_3_, #sycl.access.address_space<global>>,
// CHECK-SAME:                                                  %[[VAL_2:.*]]: memref<?x!sycl_id_3_, #sycl.access.address_space<global>>) -> memref<?x!sycl_nd_range_3_> {
// CHECK:           %[[VAL_3:.*]] = sycl.nd_range.constructor(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) : (memref<?x!sycl_range_3_, #sycl.access.address_space<global>>, memref<?x!sycl_range_3_, #sycl.access.address_space<global>>, memref<?x!sycl_id_3_, #sycl.access.address_space<global>>) -> memref<?x!sycl_nd_range_3_>
// CHECK:           return %[[VAL_3]] : memref<?x!sycl_nd_range_3_>
// CHECK:         }

func.func @nd_range_constructor_abstract_cast(
    %global_size: memref<?x!sycl_range_3_, #sycl.access.address_space<global>>,
    %local_size: memref<?x!sycl_range_3_, #sycl.access.address_space<global>>,
    %offset: memref<?x!sycl_id_3_, #sycl.access.address_space<global>>) -> memref<?x!sycl_nd_range_3_> {
  %cast0 = memref.memory_space_cast %global_size : memref<?x!sycl_range_3_, #sycl.access.address_space<global>> to memref<?x!sycl_range_3_, #sycl.access.address_space<generic>>
  %cast1 = memref.memory_space_cast %local_size : memref<?x!sycl_range_3_, #sycl.access.address_space<global>> to memref<?x!sycl_range_3_, #sycl.access.address_space<generic>>
  %cast2 = memref.memory_space_cast %offset : memref<?x!sycl_id_3_, #sycl.access.address_space<global>> to memref<?x!sycl_id_3_, #sycl.access.address_space<generic>>
  %cast2.2 = memref.cast %cast2 : memref<?x!sycl_id_3_, #sycl.access.address_space<generic>> to memref<1x!sycl_id_3_, #sycl.access.address_space<generic>>
  %res = sycl.nd_range.constructor(%cast0, %cast1, %cast2.2)
      : (memref<?x!sycl_range_3_, #sycl.access.address_space<generic>>,
         memref<?x!sycl_range_3_, #sycl.access.address_space<generic>>,
         memref<1x!sycl_id_3_, #sycl.access.address_space<generic>>)
      -> memref<?x!sycl_nd_range_3_>
  func.return %res : memref<?x!sycl_nd_range_3_>
}
