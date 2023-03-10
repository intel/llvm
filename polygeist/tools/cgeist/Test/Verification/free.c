// RUN: cgeist %s --function=* -S | FileCheck %s

void free(void*);

void metafree(void* x, void (*foo)(int), void (*bar)(void)) {
    foo(0);
    bar();
    free(x);
}

// CHECK:   func @metafree(%arg0: !llvm.ptr<i8>, %arg1: !llvm.ptr<func<void (i32)>>, %arg2: !llvm.ptr<func<void ()>>) 
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     llvm.call %arg1(%c0_i32) : !llvm.ptr<func<void (i32)>>, (i32) -> ()
// CHECK-NEXT:     llvm.call %arg2() : !llvm.ptr<func<void ()>>, () -> ()
// CHECK-NEXT:     llvm.call @free(%arg0) : (!llvm.ptr<i8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
