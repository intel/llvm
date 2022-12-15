// RUN: cgeist %s --function=* -S -O0 -w | FileCheck %s

#include <stdbool.h>

// CHECK-LABEL:   func.func @int_conversion(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32) -> i8
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.cmpi ne, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i8
// CHECK-NEXT:      return %[[VAL_3]] : i8
// CHECK-NEXT:    }
bool int_conversion(int i) { return (bool)i; }

// CHECK-LABEL:   func.func @long_conversion(
// CHECK-SAME:                               %[[VAL_0:.*]]: i64) -> i8
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.cmpi ne, %[[VAL_0]], %[[VAL_1]] : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i8
// CHECK-NEXT:      return %[[VAL_3]] : i8
// CHECK-NEXT:    }
bool long_conversion(long i) { return (bool)i; }

// CHECK-LABEL:   func.func @float_conversion(
// CHECK-SAME:                                %[[VAL_0:.*]]: f32) -> i8
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.cmpf une, %[[VAL_0]], %[[VAL_1]] : f32
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i8
// CHECK-NEXT:      return %[[VAL_3]] : i8
// CHECK-NEXT:    }
bool float_conversion(float i) { return (bool)i; }

// CHECK-LABEL:   func.func @double_conversion(
// CHECK-SAME:                                 %[[VAL_0:.*]]: f64) -> i8
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.cmpf une, %[[VAL_0]], %[[VAL_1]] : f64
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i8
// CHECK-NEXT:      return %[[VAL_3]] : i8
// CHECK-NEXT:    }
bool double_conversion(double i) { return (bool)i; }

// CHECK-LABEL:   func.func @ptr_conversion(
// CHECK-SAME:                              %[[VAL_0:.*]]: !llvm.ptr<i8>) -> i8
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.icmp "ne" %[[VAL_0]], %[[VAL_1]] : !llvm.ptr<i8>
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.extui %[[VAL_2]] : i1 to i8
// CHECK-NEXT:      return %[[VAL_3]] : i8
// CHECK-NEXT:    }
bool ptr_conversion(void *i) { return (bool)i; }

// CHECK-LABEL:   func.func @memref_conversion(
// CHECK-SAME:                                 %[[VAL_0:.*]]: memref<?xi32>) -> i8
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.null : !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_2:.*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?xi32>) -> !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.icmp "ne" %[[VAL_2]], %[[VAL_1]] : !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.extui %[[VAL_3]] : i1 to i8
// CHECK-NEXT:      return %[[VAL_4]] : i8
// CHECK-NEXT:    }
bool memref_conversion(int *i) { return (bool)i; }

// CHECK-LABEL:   func.func @bool2unsigned_conversion(
// CHECK-SAME:                                        %[[VAL_0:.*]]: i8) -> i32
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.extui %[[VAL_0]] : i8 to i32
// CHECK-NEXT:      return %[[VAL_1]] : i32
// CHECK-NEXT:    }
unsigned bool2unsigned_conversion(bool i) { return (unsigned)i; }

// CHECK-LABEL:   func.func @bool2long_conversion(
// CHECK-SAME:                                    %[[VAL_0:.*]]: i8) -> i64
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.extui %[[VAL_0]] : i8 to i64
// CHECK-NEXT:      return %[[VAL_1]] : i64
// CHECK-NEXT:    }
long bool2long_conversion(bool i) { return (long)i; }
