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
