// RUN: cgeist -O0 -w %s --function=* -S | FileCheck %s

struct N {
    int a;
    int b;
};

void copy(struct N* dst, void* src) {
    __builtin_memcpy(dst, src, sizeof(struct N));
}

// CHECK:   func @copy(%arg0: !llvm.ptr<struct<(i32, i32)>>, %arg1: !llvm.ptr<i8>) attributes {llvm.linkage = #llvm.linkage<external>}
// CHECK-DAG:      %false = arith.constant false
// CHECK-DAG:      %c8_i64 = arith.constant 8 : i64
// CHECK-DAG:      %0 = llvm.bitcast %arg0 : !llvm.ptr<struct<(i32, i32)>> to !llvm.ptr<i8>
// CHECK-NEXT:     "llvm.intr.memcpy"(%0, %arg1, %c8_i64, %false) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }

