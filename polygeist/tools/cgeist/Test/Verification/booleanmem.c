// RUN: cgeist %s -w -O0 --function=* -S | FileCheck %s
// RUN: cgeist %s -w -O0 --memref-fullrank --function=* -S | FileCheck %s --check-prefix=CHECK-FULLRANK

#include <stdbool.h>
#include <stdlib.h>

// CHECK-LABEL:    memref.global "private" @arr_no_init : memref<2xi8> = dense<0> {alignment = 1 : i64}
// CHECK-LABEL:    memref.global "private" @arr : memref<2xi8> = dense<[-1, 0]> {alignment = 1 : i64}
// CHECK-LABEL:    memref.global "private" @"static_arr_local_no_init@static@arr" : memref<10xi8> = dense<0> {alignment = 1 : i64}
// CHECK-LABEL:    memref.global "private" @"static_arr_local@static@arr@init" : memref<i1> = dense<true>
// CHECK-LABEL:    memref.global "private" @"static_arr_local@static@arr" : memref<2xi8> = dense<[-1, 0]> {alignment = 1 : i64}
// CHECK-LABEL:    memref.global "private" @x_no_init : memref<i8> = dense<0> {alignment = 1 : i64}
// CHECK-LABEL:    memref.global "private" @x : memref<i8> = dense<0> {alignment = 1 : i64}
// CHECK-LABEL:    memref.global "private" @"get_local_static@static@x@init" : memref<i1> = dense<true>
// CHECK-LABEL:    memref.global "private" @"get_local_static@static@x" : memref<i8> = dense<1> {alignment = 1 : i64}

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

bool ptr_get(bool *arr, size_t i) {
  return arr[i];
}

struct foo {
  int dummy;
  bool x;
};

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
bool struct_ptr_get(struct foo *s) {
  return s->x;
}

void keep(bool *);

// CHECK-LABEL:   func.func @get_addr()
// CHECK-NEXT:      %[[VAL_0:.*]] = arith.constant 0 : i8
// CHECK-NEXT:      %[[VAL_1:.*]] = memref.alloca() : memref<1xi8>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.undef : i8
// CHECK-NEXT:      affine.store %[[VAL_0]], %[[VAL_1]][0] : memref<1xi8>
// CHECK-NEXT:      %[[VAL_3:.*]] = memref.cast %[[VAL_1]] : memref<1xi8> to memref<?xi8>
// CHECK-NEXT:      call @keep(%[[VAL_3]]) : (memref<?xi8>) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }
void get_addr() {
  bool a = false;
  keep(&a);
}

// CHECK-LABEL:   func.func @decay_to_ptr()
// CHECK-NEXT:      %[[VAL_0:.*]] = arith.constant 0 : i8
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 1 : i8
// CHECK-NEXT:      %[[VAL_2:.*]] = memref.alloca() : memref<2xi8>
// CHECK-NEXT:      affine.store %[[VAL_1]], %[[VAL_2]][0] : memref<2xi8>
// CHECK-NEXT:      affine.store %[[VAL_0]], %[[VAL_2]][1] : memref<2xi8>
// CHECK-NEXT:      %[[VAL_3:.*]] = memref.cast %[[VAL_2]] : memref<2xi8> to memref<?xi8>
// CHECK-NEXT:      call @keep(%[[VAL_3]]) : (memref<?xi8>) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }
void decay_to_ptr() {
  bool a[2] = {true, false};
  keep(a);
}

// CHECK-LABEL:   func.func @get_local_static() -> i1
// CHECK-NEXT:      %[[VAL_0:.*]] = arith.constant 1 : i8
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant false
// CHECK-NEXT:      %[[VAL_2:.*]] = memref.get_global @"get_local_static@static@x" : memref<i8>
// CHECK-NEXT:      %[[VAL_3:.*]] = memref.alloca() : memref<1xindex>
// CHECK-NEXT:      %[[VAL_4:.*]] = memref.reshape %[[VAL_2]](%[[VAL_3]]) : (memref<i8>, memref<1xindex>) -> memref<1xi8>
// CHECK-NEXT:      %[[VAL_5:.*]] = memref.get_global @"get_local_static@static@x@init" : memref<i1>
// CHECK-NEXT:      %[[VAL_6:.*]] = memref.alloca() : memref<1xindex>
// CHECK-NEXT:      %[[VAL_7:.*]] = memref.reshape %[[VAL_5]](%[[VAL_6]]) : (memref<i1>, memref<1xindex>) -> memref<1xi1>
// CHECK-NEXT:      %[[VAL_8:.*]] = affine.load %[[VAL_7]][0] : memref<1xi1>
// CHECK-NEXT:      scf.if %[[VAL_8]] {
// CHECK-NEXT:        affine.store %[[VAL_1]], %[[VAL_7]][0] : memref<1xi1>
// CHECK-NEXT:        affine.store %[[VAL_0]], %[[VAL_4]][0] : memref<1xi8>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[VAL_9:.*]] = affine.load %[[VAL_4]][0] : memref<1xi8>
// CHECK-NEXT:      %[[VAL_10:.*]] = arith.trunci %[[VAL_9]] : i8 to i1
// CHECK-NEXT:      return %[[VAL_10]] : i1
// CHECK-NEXT:    }
bool get_local_static() {
  static bool x = true;
  return x;
}

static bool x = false;

// CHECK-LABEL:   func.func @get_global_static() -> i1
// CHECK-NEXT:      %[[VAL_0:.*]] = memref.get_global @x : memref<i8>
// CHECK-NEXT:      %[[VAL_1:.*]] = memref.alloca() : memref<1xindex>
// CHECK-NEXT:      %[[VAL_2:.*]] = memref.reshape %[[VAL_0]](%[[VAL_1]]) : (memref<i8>, memref<1xindex>) -> memref<1xi8>
// CHECK-NEXT:      %[[VAL_3:.*]] = affine.load %[[VAL_2]][0] : memref<1xi8>
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.trunci %[[VAL_3]] : i8 to i1
// CHECK-NEXT:      return %[[VAL_4]] : i1
// CHECK-NEXT:    }
bool get_global_static() {
  return x;
}

static bool x_no_init;

// CHECK-LABEL:   func.func @get_global_static_no_init() -> i1
// CHECK-NEXT:      %[[VAL_0:.*]] = memref.get_global @x_no_init : memref<i8>
// CHECK-NEXT:      %[[VAL_1:.*]] = memref.alloca() : memref<1xindex>
// CHECK-NEXT:      %[[VAL_2:.*]] = memref.reshape %[[VAL_0]](%[[VAL_1]]) : (memref<i8>, memref<1xindex>) -> memref<1xi8>
// CHECK-NEXT:      %[[VAL_3:.*]] = affine.load %[[VAL_2]][0] : memref<1xi8>
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.trunci %[[VAL_3]] : i8 to i1
// CHECK-NEXT:      return %[[VAL_4]] : i1
// CHECK-NEXT:    }
bool get_global_static_no_init() {
  return x_no_init;
}

// CHECK-LABEL:   func.func @static_arr_local()
// CHECK-NEXT:      %[[VAL_0:.*]] = arith.constant 0 : i8
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 1 : i8
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant false
// CHECK-NEXT:      %[[VAL_3:.*]] = memref.get_global @"static_arr_local@static@arr" : memref<2xi8>
// CHECK-NEXT:      %[[VAL_4:.*]] = memref.get_global @"static_arr_local@static@arr@init" : memref<i1>
// CHECK-NEXT:      %[[VAL_5:.*]] = memref.alloca() : memref<1xindex>
// CHECK-NEXT:      %[[VAL_6:.*]] = memref.reshape %[[VAL_4]](%[[VAL_5]]) : (memref<i1>, memref<1xindex>) -> memref<1xi1>
// CHECK-NEXT:      %[[VAL_7:.*]] = affine.load %[[VAL_6]][0] : memref<1xi1>
// CHECK-NEXT:      scf.if %[[VAL_7]] {
// CHECK-NEXT:        affine.store %[[VAL_2]], %[[VAL_6]][0] : memref<1xi1>
// CHECK-NEXT:        affine.store %[[VAL_1]], %[[VAL_3]][0] : memref<2xi8>
// CHECK-NEXT:        affine.store %[[VAL_0]], %[[VAL_3]][1] : memref<2xi8>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[VAL_8:.*]] = memref.cast %[[VAL_3]] : memref<2xi8> to memref<?xi8>
// CHECK-NEXT:      call @keep(%[[VAL_8]]) : (memref<?xi8>) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }
void static_arr_local() {
  static bool arr[2] = {true, false};
  keep(arr);
}

// CHECK-LABEL:   func.func @static_arr_local_no_init()
// CHECK-NEXT:      %[[VAL_0:.*]] = memref.get_global @"static_arr_local_no_init@static@arr" : memref<10xi8>
// CHECK-NEXT:      %[[VAL_1:.*]] = memref.cast %[[VAL_0]] : memref<10xi8> to memref<?xi8>
// CHECK-NEXT:      call @keep(%[[VAL_1]]) : (memref<?xi8>) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }
void static_arr_local_no_init() {
  static bool arr[10];
  keep(arr);
}

static bool arr[2] = {true, false};

// CHECK-LABEL:   func.func @static_arr_global() -> i1
// CHECK-NEXT:      %[[VAL_0:.*]] = memref.get_global @arr : memref<2xi8>
// CHECK-NEXT:      %[[VAL_1:.*]] = affine.load %[[VAL_0]][0] : memref<2xi8>
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.trunci %[[VAL_1]] : i8 to i1
// CHECK-NEXT:      return %[[VAL_2]] : i1
// CHECK-NEXT:    }
bool static_arr_global() {
  return arr[0];
}

static bool arr_no_init[2];

// CHECK-LABEL:   func.func @static_arr_global_no_init() -> i1
// CHECK-NEXT:      %[[VAL_0:.*]] = memref.get_global @arr_no_init : memref<2xi8>
// CHECK-NEXT:      %[[VAL_1:.*]] = affine.load %[[VAL_0]][0] : memref<2xi8>
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.trunci %[[VAL_1]] : i8 to i1
// CHECK-NEXT:      return %[[VAL_2]] : i1
// CHECK-NEXT:    }
bool static_arr_global_no_init() {
  return arr_no_init[0];
}
