// RUN: cgeist %s -O2 --function=fp32_from_bits -S | FileCheck %s

#include <stdint.h>
float fp32_from_bits(uint32_t w) {
    union {
      uint32_t as_bits;
      float as_value;
    } fp32 = {w};
    return fp32.as_value;
}

// CHECK-LABEL:   func.func @fp32_from_bits(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(i32)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32)>
// CHECK-NEXT:      llvm.store %[[VAL_0]], %[[VAL_3]] : i32, !llvm.ptr
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> f32
// CHECK-NEXT:      return %[[VAL_4]] : f32
// CHECK-NEXT:    }
