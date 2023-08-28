// RUN: cgeist  %s --function=* -S | FileCheck %s

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

// CHECK-LABEL:   func.func @_Z3refR1B(
// CHECK-SAME:                         %[[VAL_0:.*]]: !llvm.ptr) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i32, f64)>, ptr)>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i32
// CHECK-NEXT:      return %[[VAL_3]] : i32
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_Z3ptrP1B(
// CHECK-SAME:                         %[[VAL_0:.*]]: !llvm.ptr) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i32, f64)>, ptr)>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f64)>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i32
// CHECK-NEXT:      return %[[VAL_3]] : i32
// CHECK-NEXT:    }
