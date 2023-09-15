// RUN: cgeist %s --function=* -S -O0 -w | FileCheck %s

#include <stdint.h>
#include <stdlib.h>

// CHECK-LABEL:   func.func @ptr2size_t(
// CHECK-SAME:                          %[[VAL_0:.*]]: !llvm.ptr) -> i64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.ptrtoint %[[VAL_0]] : !llvm.ptr to i64
// CHECK-NEXT:      return %[[VAL_1]] : i64
// CHECK-NEXT:    }
size_t ptr2size_t(void * i) { return (size_t)i; }

// CHECK-LABEL:   func.func @ptr2int8_t(
// CHECK-SAME:                          %[[VAL_0:.*]]: !llvm.ptr) -> i8 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.ptrtoint %[[VAL_0]] : !llvm.ptr to i8
// CHECK-NEXT:      return %[[VAL_1]] : i8
// CHECK-NEXT:    }
int8_t ptr2int8_t(void * i) { return (int8_t)i; }

// CHECK-LABEL:   func.func @memref2size_t(
// CHECK-SAME:                             %[[VAL_0:.*]]: memref<?xi32>) -> i64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?xi32>) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.ptrtoint %[[VAL_1]] : !llvm.ptr to i64
// CHECK-NEXT:      return %[[VAL_2]] : i64
// CHECK-NEXT:    }
size_t memref2size_t(int * i) { return (size_t)i; }

// CHECK-LABEL:   func.func @memref2int8_t(
// CHECK-SAME:                             %[[VAL_0:.*]]: memref<?xi32>) -> i8 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?xi32>) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.ptrtoint %[[VAL_1]] : !llvm.ptr to i8
// CHECK-NEXT:      return %[[VAL_2]] : i8
// CHECK-NEXT:    }
uint8_t memref2int8_t(int * i) { return (uint8_t)i; }
