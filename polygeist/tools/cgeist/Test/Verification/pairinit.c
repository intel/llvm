// RUN: cgeist -O0 -w %s --function=func -S | FileCheck %s

struct pair {
    int x, y;
};

struct pair func() {
    struct pair tmp = {2, 3};
    return tmp;
}

// CHECK-LABEL:   func.func @func() -> !llvm.struct<(i32, i32)> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 3 : i32
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<(i32, i32)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_3]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
// CHECK-NEXT:      llvm.store %[[VAL_1]], %[[VAL_4]] : i32, !llvm.ptr
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_3]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
// CHECK-NEXT:      llvm.store %[[VAL_0]], %[[VAL_5]] : i32, !llvm.ptr
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> !llvm.struct<(i32, i32)>
// CHECK-NEXT:      return %[[VAL_6]] : !llvm.struct<(i32, i32)>
// CHECK-NEXT:    }
