// RUN: cgeist -O0 -w %s --function=func -S | FileCheck %s

struct pair {
    int x, y;
};

struct pair func() {
    struct pair tmp = {2, 3};
    return tmp;
}

// CHECK:    func.func @func() -> !llvm.struct<(i32, i32)> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %c3_i32 = arith.constant 3 : i32
// CHECK-DAG:     %c2_i32 = arith.constant 2 : i32
// CHECK-DAG:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i32, i32)> : (i64) -> !llvm.ptr<struct<(i32, i32)>>
// CHECK-NEXT:    %1 = llvm.getelementptr inbounds %0[0, 0] : (!llvm.ptr<struct<(i32, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:    llvm.store %c2_i32, %1 : !llvm.ptr<i32>
// CHECK-NEXT:    %2 = llvm.getelementptr inbounds %0[0, 1] : (!llvm.ptr<struct<(i32, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:    llvm.store %c3_i32, %2 : !llvm.ptr<i32>
// CHECK-NEXT:    %3 = llvm.load %0 : !llvm.ptr<struct<(i32, i32)>>
// CHECK-NEXT:    return %3 : !llvm.struct<(i32, i32)>
