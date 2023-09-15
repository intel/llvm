// RUN: cgeist %s --function=* -S | FileCheck %s

void free(void*);

void metafree(void* x, void (*foo)(int), void (*bar)(void)) {
    foo(0);
    bar();
    free(x);
}


// CHECK:      func.func @metafree(%[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: !llvm.ptr, %[[VAL_2:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      llvm.call %[[VAL_1]](%[[VAL_3]]) : !llvm.ptr, (i32) -> ()
// CHECK-NEXT:      llvm.call %[[VAL_2]]() : !llvm.ptr, () -> ()
// CHECK-NEXT:      llvm.call @free(%[[VAL_0]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }
