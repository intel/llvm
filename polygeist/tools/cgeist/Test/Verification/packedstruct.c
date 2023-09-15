// RUN: cgeist %s --function=* -S | FileCheck %s

struct meta {
    long long a;
    char dtype;
};

struct fin {
    struct meta f;
    char dtype;
} __attribute__((packed)) ;

long long run(struct meta m, char c);

void compute(struct fin f) {
    run(f.f, f.dtype);
}

// CHECK-LABEL:   func.func @compute(
// CHECK-SAME:                       %[[VAL_0:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i64, i8)>, i8)>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(i64, i8)>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i64, i8)>, i8)>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i8
// CHECK-NEXT:      %[[VAL_5:.*]] = call @run(%[[VAL_2]], %[[VAL_4]]) : (!llvm.struct<(i64, i8)>, i8) -> i64
// CHECK-NEXT:      return
// CHECK-NEXT:    }

