#include <cstdint>

// RUN: cgeist  -O0 %s --function=* -S | FileCheck %s

// CHECK-LABEL:   func.func @_Z17reinterpret_floatRf(
// CHECK-SAME:                                       %[[VAL_0:.*]]: memref<?xf32>) -> memref<?xi32>
// CHECK:           %[[VAL_1:.*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?xf32>) -> !llvm.ptr
// CHECK:           %[[VAL_2:.*]] = "polygeist.pointer2memref"(%[[VAL_1]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK:           return %[[VAL_2]] : memref<?xi32>
// CHECK:         }

int32_t &reinterpret_float(float &f) {
  return reinterpret_cast<int32_t &>(f);
}

// CHECK-LABEL:   func.func @_Z17reinterpret_floatf(
// CHECK-SAME:                                      %[[VAL_0:.*]]: f32) -> i32
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<1xf32>
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.undef : f32
// CHECK:           affine.store %[[VAL_0]], %[[VAL_1]][0] : memref<1xf32>
// CHECK:           %[[VAL_3:.*]] = "polygeist.memref2pointer"(%[[VAL_1]]) : (memref<1xf32>) -> !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i32
// CHECK:           return %[[VAL_4]] : i32
// CHECK:         }

int32_t reinterpret_float(float f) {
  return reinterpret_cast<int32_t &>(f);
}

struct fooint32 { int32_t x; };
struct foofloat { float x; };

// CHECK-LABEL:   func.func @_Z17reinterpret_floatR8foofloat(
// CHECK-SAME:                                               %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.ptr
// CHECK:           return %[[VAL_0]] : !llvm.ptr
// CHECK:         }

fooint32 &reinterpret_float(foofloat &f) {
  return reinterpret_cast<fooint32 &>(f);
}

// COM: Copy constructor is called

// CHECK-LABEL:   func.func @_Z17reinterpret_float8foofloat(
// CHECK-SAME:                                              %[[VAL_0:.*]]: !llvm.struct<(f32)>) -> !llvm.struct<(i32)>
// CHECK:           %[[SIZE:.*]] = arith.constant 4 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(i32)> : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(f32)> : (i64) -> !llvm.ptr
// CHECK:           llvm.store %[[VAL_0]], %[[VAL_3]] : !llvm.struct<(f32)>, !llvm.ptr
// CHECK:           "llvm.intr.memcpy"(%[[VAL_2]], %[[VAL_3]], %[[SIZE]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> !llvm.struct<(i32)>
// CHECK:           return %[[VAL_4]] : !llvm.struct<(i32)>
// CHECK:         }

fooint32 reinterpret_float(foofloat f) {
  return reinterpret_cast<fooint32 &>(f);
}
