// RUN: cgeist %s --function=* -S | FileCheck %s

struct A {
    int x;
    double y;
};

struct B : public A {
    void* z;
};

int ref(struct B& v) {
    return v.x;
}

int ptr(struct B* v) {
    return v->x;
}

// CHECK:   func @_Z3refR1B(%arg0: !llvm.ptr<struct<(struct<(i32, f64)>, ptr<i8>)>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = llvm.getelementptr %arg0[%c0_i32, 0] : (!llvm.ptr<struct<(struct<(i32, f64)>, ptr<i8>)>>, i32) -> !llvm.ptr<struct<(i32, f64)>>
// CHECK-NEXT:     %1 = llvm.getelementptr %0[%c0_i32, 0] : (!llvm.ptr<struct<(i32, f64)>>, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:     %2 = llvm.load %1 : !llvm.ptr<i32>
// CHECK-NEXT:     return %2 : i32
// CHECK-NEXT:   }
// CHECK:   func @_Z3ptrP1B(%arg0: !llvm.ptr<struct<(struct<(i32, f64)>, ptr<i8>)>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = llvm.getelementptr %arg0[%c0_i32, 0] : (!llvm.ptr<struct<(struct<(i32, f64)>, ptr<i8>)>>, i32) -> !llvm.ptr<struct<(i32, f64)>>
// CHECK-NEXT:     %1 = llvm.getelementptr %0[%c0_i32, 0] : (!llvm.ptr<struct<(i32, f64)>>, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:     %2 = llvm.load %1 : !llvm.ptr<i32>
// CHECK-NEXT:     return %2 : i32
// CHECK-NEXT:   }
