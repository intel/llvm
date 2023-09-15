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

// CHECK-LABEL:   func.func @byval(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                     %[[VAL_1:.*]]: i32) -> !llvm.struct<(i32, i32)> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_0]] : !llvm.ptr -> !llvm.struct<(i32, i32)>
// CHECK-NEXT:      return %[[VAL_2]] : !llvm.struct<(i32, i32)>
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @create() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<(i32, i32)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<(i32, i32)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_5]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
// CHECK-NEXT:      llvm.store %[[VAL_2]], %[[VAL_6]] : i32, !llvm.ptr
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.getelementptr inbounds %[[VAL_5]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
// CHECK-NEXT:      llvm.store %[[VAL_1]], %[[VAL_7]] : i32, !llvm.ptr
// CHECK-NEXT:      %[[VAL_8:.*]] = call @byval0(%[[VAL_5]], %[[VAL_0]]) : (!llvm.ptr, i32) -> !llvm.struct<(i32, i32)>
// CHECK-NEXT:      llvm.store %[[VAL_8]], %[[VAL_4]] : !llvm.struct<(i32, i32)>, !llvm.ptr
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_4]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr -> i32
// CHECK-NEXT:      return %[[VAL_10]] : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @byval0(!llvm.ptr, i32) -> !llvm.struct<(i32, i32)> attributes {llvm.linkage = #llvm.linkage<external>}
