// RUN: cgeist %s -w -O0 --function=* -S | FileCheck %s
// RUN: cgeist %s -w -O0 --memref-fullrank --function=* -S | FileCheck %s --check-prefix=CHECK-FULLRANK

#include <stdbool.h>
#include <stdlib.h>

struct foo {
  int dummy;
  bool x;
};

// CHECK-LABEL:   func.func @array_get(
// CHECK-SAME:                         %[[VAL_0:.*]]: memref<?xi8>,
// CHECK-SAME:                         %[[VAL_1:.*]]: i64) -> i1
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK-NEXT:      %[[VAL_3:.*]] = affine.load %[[VAL_0]][symbol(%[[VAL_2]])] : memref<?xi8>
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.trunci %[[VAL_3]] : i8 to i1
// CHECK-NEXT:      return %[[VAL_4]] : i1
// CHECK-NEXT:    }

// CHECK-FULLRANK-LABEL:   func.func @array_get(
// CHECK-FULLRANK-SAME:                         %[[VAL_0:.*]]: memref<100xi8>,
// CHECK-FULLRANK-SAME:                         %[[VAL_1:.*]]: i64) -> i1
// CHECK-FULLRANK-NEXT:      %[[VAL_2:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK-FULLRANK-NEXT:      %[[VAL_3:.*]] = affine.load %[[VAL_0]][symbol(%[[VAL_2]])] : memref<100xi8>
// CHECK-FULLRANK-NEXT:      %[[VAL_4:.*]] = arith.trunci %[[VAL_3]] : i8 to i1
// CHECK-FULLRANK-NEXT:      return %[[VAL_4]] : i1
// CHECK-FULLRANK-NEXT:    }
bool array_get(bool arr[100], size_t i) {
  return arr[i];
}

// CHECK-LABEL:   func.func @array_2d_get(
// CHECK-SAME:                            %[[VAL_0:.*]]: memref<?x100xi8>,
// CHECK-SAME:                            %[[VAL_1:.*]]: i64,
// CHECK-SAME:                            %[[VAL_2:.*]]: i64) -> i1
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.index_cast %[[VAL_2]] : i64 to index
// CHECK-NEXT:      %[[VAL_5:.*]] = affine.load %[[VAL_0]][symbol(%[[VAL_3]]), symbol(%[[VAL_4]])] : memref<?x100xi8>
// CHECK-NEXT:      %[[VAL_6:.*]] = arith.trunci %[[VAL_5]] : i8 to i1
// CHECK-NEXT:      return %[[VAL_6]] : i1
// CHECK-NEXT:    }

// CHECK-FULLRANK-LABEL:   func.func @array_2d_get(
// CHECK-FULLRANK-SAME:                            %[[VAL_0:.*]]: memref<100x100xi8>,
// CHECK-FULLRANK-SAME:                            %[[VAL_1:.*]]: i64,
// CHECK-FULLRANK-SAME:                            %[[VAL_2:.*]]: i64) -> i1
// CHECK-FULLRANK-NEXT:      %[[VAL_3:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK-FULLRANK-NEXT:      %[[VAL_4:.*]] = arith.index_cast %[[VAL_2]] : i64 to index
// CHECK-FULLRANK-NEXT:      %[[VAL_5:.*]] = affine.load %[[VAL_0]][symbol(%[[VAL_3]]), symbol(%[[VAL_4]])] : memref<100x100xi8>
// CHECK-FULLRANK-NEXT:      %[[VAL_6:.*]] = arith.trunci %[[VAL_5]] : i8 to i1
// CHECK-FULLRANK-NEXT:      return %[[VAL_6]] : i1
// CHECK-FULLRANK-NEXT:    }
bool array_2d_get(bool arr[100][100], size_t i, size_t j) {
  return arr[i][j];
}

// CHECK-LABEL:   func.func @ptr_get(
// CHECK-SAME:                       %[[VAL_0:.*]]: memref<?xi8>,
// CHECK-SAME:                       %[[VAL_1:.*]]: i64) -> i1
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK-NEXT:      %[[VAL_3:.*]] = affine.load %[[VAL_0]][symbol(%[[VAL_2]])] : memref<?xi8>
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.trunci %[[VAL_3]] : i8 to i1
// CHECK-NEXT:      return %[[VAL_4]] : i1
// CHECK-NEXT:    }

// CHECK-FULLRANK-LABEL:   func.func @ptr_get(
// CHECK-FULLRANK-SAME:                       %[[VAL_0:.*]]: memref<?xi8>,
// CHECK-FULLRANK-SAME:                       %[[VAL_1:.*]]: i64) -> i1
// CHECK-FULLRANK-NEXT:      %[[VAL_2:.*]] = arith.index_cast %[[VAL_1]] : i64 to index
// CHECK-FULLRANK-NEXT:      %[[VAL_3:.*]] = affine.load %[[VAL_0]][symbol(%[[VAL_2]])] : memref<?xi8>
// CHECK-FULLRANK-NEXT:      %[[VAL_4:.*]] = arith.trunci %[[VAL_3]] : i8 to i1
// CHECK-FULLRANK-NEXT:      return %[[VAL_4]] : i1
// CHECK-FULLRANK-NEXT:    }
bool ptr_get(bool *arr, size_t i) {
  return arr[i];
}

// CHECK-LABEL:   func.func @struct_get(
// CHECK-SAME:                          %[[VAL_0:.*]]: !llvm.struct<(i32, i8)>) -> i1
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(i32, i8)> : (i64) -> !llvm.ptr<struct<(i32, i8)>>
// CHECK-NEXT:      llvm.store %[[VAL_0]], %[[VAL_2]] : !llvm.ptr<struct<(i32, i8)>>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_2]][0, 1] : (!llvm.ptr<struct<(i32, i8)>>) -> !llvm.ptr<i8>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i8>
// CHECK-NEXT:      %[[VAL_5:.*]] = arith.trunci %[[VAL_4]] : i8 to i1
// CHECK-NEXT:      return %[[VAL_5]] : i1
// CHECK-NEXT:    }

// CHECK-FULLRANK-LABEL:   func.func @struct_get(
// CHECK-FULLRANK-SAME:                          %[[VAL_0:.*]]: !llvm.struct<(i32, i8)>) -> i1
// CHECK-FULLRANK-NEXT:      %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-FULLRANK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(i32, i8)> : (i64) -> !llvm.ptr<struct<(i32, i8)>>
// CHECK-FULLRANK-NEXT:      llvm.store %[[VAL_0]], %[[VAL_2]] : !llvm.ptr<struct<(i32, i8)>>
// CHECK-FULLRANK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_2]][0, 1] : (!llvm.ptr<struct<(i32, i8)>>) -> !llvm.ptr<i8>
// CHECK-FULLRANK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i8>
// CHECK-FULLRANK-NEXT:      %[[VAL_5:.*]] = arith.trunci %[[VAL_4]] : i8 to i1
// CHECK-FULLRANK-NEXT:      return %[[VAL_5]] : i1
// CHECK-FULLRANK-NEXT:    }
bool struct_get(struct foo s) {
  return s.x;
}

// CHECK-LABEL:   func.func @struct_ptr_get(
// CHECK-SAME:                              %[[VAL_0:.*]]: !llvm.ptr<struct<(i32, i8)>>) -> i1
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr %[[VAL_0]][0, 1] : (!llvm.ptr<struct<(i32, i8)>>) -> !llvm.ptr<i8>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i8>
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.trunci %[[VAL_2]] : i8 to i1
// CHECK-NEXT:      return %[[VAL_3]] : i1
// CHECK-NEXT:    }

// CHECK-FULLRANK-LABEL:   func.func @struct_ptr_get(
// CHECK-FULLRANK-SAME:                              %[[VAL_0:.*]]: !llvm.ptr<struct<(i32, i8)>>) -> i1
// CHECK-FULLRANK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr %[[VAL_0]][0, 1] : (!llvm.ptr<struct<(i32, i8)>>) -> !llvm.ptr<i8>
// CHECK-FULLRANK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<i8>
// CHECK-FULLRANK-NEXT:      %[[VAL_3:.*]] = arith.trunci %[[VAL_2]] : i8 to i1
// CHECK-FULLRANK-NEXT:      return %[[VAL_3]] : i1
// CHECK-FULLRANK-NEXT:    }
bool struct_ptr_get(struct foo *s) {
  return s->x;
}
