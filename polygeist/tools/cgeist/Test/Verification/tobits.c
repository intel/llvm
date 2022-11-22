// RUN: cgeist %s -O2 --function=fp32_from_bits -S | FileCheck %s

#include <stdint.h>
float fp32_from_bits(uint32_t w) {
    union {
      uint32_t as_bits;
      float as_value;
    } fp32 = {w};
    return fp32.as_value;
}

// CHECK:   func @fp32_from_bits(%arg0: i32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<(i32)> : (i64) -> !llvm.ptr<struct<(i32)>>
// CHECK-NEXT:     %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr<struct<(i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %arg0, %1 : !llvm.ptr<i32>
// CHECK-NEXT:     %2 = llvm.bitcast %1 : !llvm.ptr<i32> to !llvm.ptr<f32>
// CHECK-NEXT:     %3 = llvm.load %2 : !llvm.ptr<f32>
// CHECK-NEXT:     return %3 : f32
// CHECK-NEXT:   }
