// RUN: cgeist %s -O0 --function=* -S | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: cgeist %s -O0 --function=* -S -emit-llvm | FileCheck %s --check-prefix=CHECK-LLVM
// XFAIL: *

// Our type generation currently does not differ between scalar and memory
// representation. Here we are generating the memory representation of union
// int_wrapper instead of its scalar representation. This test will thus fail.

// Also see issue: https://github.com/intel/llvm/issues/7994

union int_wrapper {
  int *ptr;
};

// CHECK-MLIR-LABEL:   func.func @foo(
// CHECK-MLIR-SAME:                   %[[VAL_0:.*]]: memref<?xi32>) -> memref<?xi32>
// CHECK-MLIR-NEXT:      return %[[VAL_0]] : memref<?xi32>
// CHECK-MLIR-NEXT:    }

// CHECK-LLVM-LABEL:   define i32* @foo(
// CHECK-LLVM-SAME:                     i32* %[[VAL_0:.*]]) {
// CHECK-LLVM-NEXT:      ret i32* %[[VAL_0]]
// CHECK-LLVM-NEXT:    }

union int_wrapper foo(union int_wrapper w) {
  return w;
}
