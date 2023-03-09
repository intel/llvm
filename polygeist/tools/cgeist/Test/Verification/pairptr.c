// RUN: cgeist -O0 -w %s --function=* -S | FileCheck %s

typedef struct {
  int a, b;
} pair;

pair byval0(pair* a, int x);
pair byval(pair* a, int x) {
  return *a;
}

int create() {
  pair p;
  p.a = 0;
  p.b = 1;
  pair p2 = byval0(&p, 2);
  return p2.a;
}

// CHECK:     func.func @byval(%arg0: !llvm.ptr<struct<(i32, i32)>>, %arg1: i32) -> !llvm.struct<(i32, i32)> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %0 = llvm.load %arg0 : !llvm.ptr<struct<(i32, i32)>>
// CHECK-NEXT:    return %0 : !llvm.struct<(i32, i32)>
// CHECK-NEXT:  }
// CHECK:   func @create() -> i32
// CHECK-DAG:     %c2_i32 = arith.constant 2 : i32
// CHECK-DAG:     %c1_i32 = arith.constant 1 : i32
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-DAG:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i32, i32)> : (i64) -> !llvm.ptr<struct<(i32, i32)>>
// CHECK-NEXT:    %1 = llvm.alloca %c1_i64 x !llvm.struct<(i32, i32)> : (i64) -> !llvm.ptr<struct<(i32, i32)>>
// CHECK-NEXT:    %2 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr<struct<(i32, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:    llvm.store %c0_i32, %2 : !llvm.ptr<i32>
// CHECK-NEXT:    %3 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr<struct<(i32, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:    llvm.store %c1_i32, %3 : !llvm.ptr<i32>
// CHECK-NEXT:    %4 = call @byval0(%1, %c2_i32) : (!llvm.ptr<struct<(i32, i32)>>, i32) -> !llvm.struct<(i32, i32)>
// CHECK-NEXT:    llvm.store %4, %0 : !llvm.ptr<struct<(i32, i32)>>
// CHECK-NEXT:    %5 = llvm.getelementptr inbounds %0[0, 0] : (!llvm.ptr<struct<(i32, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:    %6 = llvm.load %5 : !llvm.ptr<i32>
// CHECK-NEXT:    return %6 : i32
// CHECK-NEXT:   }
