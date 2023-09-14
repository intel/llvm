// RUN: cgeist %s --function=* -S -O0 -w | FileCheck %s

#include <stdint.h>
#include <stdlib.h>

// CHECK-LABEL:   func.func @size_t2ptr(
// CHECK-SAME:                          %[[VAL_0:.*]]: i64) -> !llvm.ptr attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.inttoptr %[[VAL_0]] : i64 to !llvm.ptr
// CHECK-NEXT:      return %[[VAL_1]] : !llvm.ptr
// CHECK-NEXT:    }
void *size_t2ptr(size_t i) { return (void *)i; }

// CHECK-LABEL:   func.func @int8_t2ptr(
// CHECK-SAME:                          %[[VAL_0:.*]]: i8) -> !llvm.ptr attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.extsi %[[VAL_0]] : i8 to i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.inttoptr %[[VAL_1]] : i64 to !llvm.ptr
// CHECK-NEXT:      return %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:    }
void *int8_t2ptr(int8_t i) { return (void *)i; }

// CHECK-LABEL:   func.func @size_t2memref(
// CHECK-SAME:                             %[[VAL_0:.*]]: i64) -> memref<?xi32> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.inttoptr %[[VAL_0]] : i64 to !llvm.ptr
// CHECK-NEXT:      %[[VAL_2:.*]] = "polygeist.pointer2memref"(%[[VAL_1]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT:      return %[[VAL_2]] : memref<?xi32>
// CHECK-NEXT:    }
int *size_t2memref(size_t i) { return (int *)i; }

// CHECK-LABEL:   func.func @int8_t2memref(
// CHECK-SAME:                             %[[VAL_0:.*]]: i8) -> memref<?xi32> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.extui %[[VAL_0]] : i8 to i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.inttoptr %[[VAL_1]] : i64 to !llvm.ptr
// CHECK-NEXT:      %[[VAL_3:.*]] = "polygeist.pointer2memref"(%[[VAL_2]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT:      return %[[VAL_3]] : memref<?xi32>
// CHECK-NEXT:    }
int *int8_t2memref(uint8_t i) { return (int *)i; }
