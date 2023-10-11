// RUN: cgeist %s --function=* -S --raise-scf-to-affine=false | FileCheck %s

void* calloc(unsigned long a, unsigned long b);

float* zmem(int n) {
    float* out = (float*)calloc(sizeof(float), n);
    return out;
}

// CHECK-LABEL:   func.func @zmem(
// CHECK-SAME:                    %[[VAL_0:.*]]: i32) -> memref<?xf32>
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 4 : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.extsi %[[VAL_0]] : i32 to i64
// CHECK-NEXT:      %[[VAL_3:.*]] = call @calloc(%[[VAL_1]], %[[VAL_2]]) : (i64, i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_4:.*]] = "polygeist.pointer2memref"(%[[VAL_3]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:      return %[[VAL_4]] : memref<?xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @calloc(i64, i64) -> !llvm.ptr
