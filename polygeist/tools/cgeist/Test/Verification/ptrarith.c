// RUN: cgeist %s --function=* -S | FileCheck %s

#include <stddef.h>

// CHECK-LABEL:   func.func @f0(
// CHECK-SAME:                  %[[VAL_0:.*]]: memref<?xi32>,
// CHECK-SAME:                  %[[VAL_1:.*]]: i32) -> memref<?xi32>
// CHECK:           %[[INDEX:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:           %[[ADD:.*]] = "polygeist.subindex"(%[[VAL_0]], %[[INDEX]]) : (memref<?xi32>, index) -> memref<?xi32>
// CHECK:           return %[[ADD]] : memref<?xi32>
// CHECK:         }

int *f0(int *ptr, int index) {
  return ptr + index;
}

// CHECK-LABEL:   func.func @f1(
// CHECK-SAME:                  %[[VAL_0:.*]]: i64,
// CHECK-SAME:                  %[[VAL_1:.*]]: memref<?xi32>) -> memref<?xi32>
// CHECK:           %[[INDEX:.*]] = arith.index_castui %[[VAL_0]] : i64 to index
// CHECK:           %[[ADD:.*]] = "polygeist.subindex"(%[[VAL_1]], %[[INDEX]]) : (memref<?xi32>, index) -> memref<?xi32>
// CHECK:           return %[[ADD]] : memref<?xi32>
// CHECK:         }

int *f1(size_t index, int *ptr) {
  return index + ptr;
}

// CHECK-LABEL:   func.func @f2(
// CHECK-SAME:                  %[[VAL_0:.*]]: i64) -> !llvm.ptr
// CHECK:           %[[PTR_0:.*]] = llvm.inttoptr %[[VAL_0]] : i64 to !llvm.ptr
// CHECK:           return %[[PTR_0]] : !llvm.ptr
// CHECK:         }

void *f2(size_t index) {
  return ((char*) NULL) + index;
}

// CHECK-LABEL:   func.func @f3(
// CHECK-SAME:                  %[[VAL_0:.*]]: memref<?xi32>,
// CHECK-SAME:                  %[[VAL_1:.*]]: i64) -> memref<?xi32>
// CHECK:           %[[I64_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[NEG:.*]] = arith.subi %[[I64_0]], %[[VAL_1]] : i64
// CHECK:           %[[INDEX:.*]] = arith.index_castui %[[NEG]] : i64 to index
// CHECK:           %[[ADD:.*]] = "polygeist.subindex"(%[[VAL_0]], %[[INDEX]]) : (memref<?xi32>, index) -> memref<?xi32>
// CHECK:           return %[[ADD]] : memref<?xi32>
// CHECK:         }

int *f3(int *ptr, size_t index) {
  return ptr - index;
}

// CHECK-LABEL:   func.func @f4(
// CHECK-SAME:                  %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                  %[[VAL_1:.*]]: i64) -> !llvm.ptr attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK:           %[[VAL_2:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK:           return %[[VAL_2]] : !llvm.ptr
// CHECK:         }

void *f4(void *ptr, size_t index) {
  return ptr + index;
}

// CHECK-LABEL:   func.func @f5(
// CHECK-SAME:                  %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                  %[[VAL_1:.*]]: i64) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK:           %[[VAL_2:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK:           %[[VAL_3:.*]] = llvm.call %[[VAL_2]]() : !llvm.ptr, () -> i32
// CHECK:           return %[[VAL_3]] : i32
// CHECK:         }
int f5(int (*ptr)(void), size_t index) {
  return (ptr + index)();
}
