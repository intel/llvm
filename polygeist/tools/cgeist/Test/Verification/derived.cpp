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
// CHECK-NEXT:     %0 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr<struct<(struct<(i32, f64)>, ptr<i8>)>>) -> !llvm.ptr<struct<(i32, f64)>>
// CHECK-NEXT:     %1 = llvm.getelementptr inbounds %0[0, 0] : (!llvm.ptr<struct<(i32, f64)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     %2 = llvm.load %1 : !llvm.ptr<i32>
// CHECK-NEXT:     return %2 : i32
// CHECK-NEXT:   }
// CHECK:   func @_Z3ptrP1B(%arg0: !llvm.ptr<struct<(struct<(i32, f64)>, ptr<i8>)>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr<struct<(struct<(i32, f64)>, ptr<i8>)>>) -> !llvm.ptr<struct<(i32, f64)>>
// CHECK-NEXT:     %1 = llvm.getelementptr inbounds %0[0, 0] : (!llvm.ptr<struct<(i32, f64)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     %2 = llvm.load %1 : !llvm.ptr<i32>
// CHECK-NEXT:     return %2 : i32
// CHECK-NEXT:   }
