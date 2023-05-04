// RUN: cgeist --use-opaque-pointers -O0 -w %s --function=* -S | FileCheck %s

struct N {
    int a;
    int b;
};

void copy(struct N* dst, void* src) {
    __builtin_memcpy(dst, src, sizeof(struct N));
}

// CHECK-LABEL:   func.func @copy(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                    %[[VAL_1:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant 8 : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.constant false
// CHECK-NEXT:      "llvm.intr.memcpy"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }

