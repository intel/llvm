// RUN: polygeist-opt -licm %s | FileCheck %s

!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_nd_range_1_ = !sycl.nd_range<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>

// CHECK-LABEL:   func.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: memref<?x!sycl_nd_range_1_>) {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_3:.*]] = memref.alloca() : memref<1x!sycl_nd_range_1_>
// CHECK-DAG:       %[[VAL_4:.*]] = memref.alloca() : memref<1x!sycl_range_1_>
// CHECK-DAG:       %[[VAL_5:.*]] = memref.alloca() : memref<1x!sycl_range_1_>
// CHECK:           %[[VAL_6:.*]] = memref.dim %[[VAL_0]], %[[VAL_1]] : memref<?x!sycl_nd_range_1_>
// CHECK:           %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_1]], %[[VAL_6]] : index
// CHECK:           scf.if %[[VAL_7]] {
// CHECK:             sycl.constructor @range(%[[VAL_4]], %[[VAL_1]]) {MangledFunctionName = @range} : (memref<1x!sycl_range_1_>, index)
// CHECK:             sycl.constructor @range(%[[VAL_5]], %[[VAL_1]]) {MangledFunctionName = @range} : (memref<1x!sycl_range_1_>, index)
// CHECK:             sycl.constructor @nd_range(%[[VAL_3]], %[[VAL_4]], %[[VAL_5]]) {MangledFunctionName = @nd_range} : (memref<1x!sycl_nd_range_1_>, memref<1x!sycl_range_1_>, memref<1x!sycl_range_1_>)
// CHECK:             %[[VAL_8:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_1]]] : memref<1x!sycl_nd_range_1_>
// CHECK:             scf.for %[[VAL_9:.*]] = %[[VAL_1]] to %[[VAL_6]] step %[[VAL_2]] {
// CHECK:               memref.store %[[VAL_8]], %[[VAL_0]]{{\[}}%[[VAL_9]]] : memref<?x!sycl_nd_range_1_>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @test(%out: memref<?x!sycl_nd_range_1_>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %nd = memref.alloca() : memref<1x!sycl_nd_range_1_>
  %gs = memref.alloca() : memref<1x!sycl_range_1_>
  %ls = memref.alloca() : memref<1x!sycl_range_1_>
  %size = memref.dim %out, %c0 : memref<?x!sycl_nd_range_1_>
  scf.for %i = %c0 to %size step %c1 {
    sycl.constructor @range(%gs, %c0) {MangledFunctionName = @range} : (memref<1x!sycl_range_1_>, index)
    sycl.constructor @range(%ls, %c0) {MangledFunctionName = @range} : (memref<1x!sycl_range_1_>, index)
    sycl.constructor @nd_range(%nd, %gs, %ls) {MangledFunctionName = @nd_range} : (memref<1x!sycl_nd_range_1_>, memref<1x!sycl_range_1_>, memref<1x!sycl_range_1_>)
    %val = memref.load %nd[%c0] : memref<1x!sycl_nd_range_1_>
    memref.store %val, %out[%i] : memref<?x!sycl_nd_range_1_>
  }
  return
}
