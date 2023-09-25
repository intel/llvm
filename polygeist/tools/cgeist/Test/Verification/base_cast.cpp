// RUN: cgeist %s --function=* -S | FileCheck %s

struct A {
  int val1;
  int val2;
};

struct B {
  bool bool1;
};

struct C : B, A {
  int val3;
};

struct D : C {
  int val3;
  bool bool2;
};

C* castAtoC(A *a) {
  // A -> C
  return static_cast<C *>(a);
};

D* castBtoD(B *b) {
  // B -> C -> D
  return static_cast<D *>(b);
}

D* castAtoD(A *b) {
  // A -> C -> D
  return static_cast<D *>(b);
}

// CHECK-LABEL:   func.func @main() -> i32
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(struct<(struct<(i8)>, struct<(i32, i32)>, i32)>, i32, i8)> : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(struct<(i8)>, struct<(i32, i32)>, i32)> : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_3]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i8)>, struct<(i32, i32)>, i32)>
// CHECK:           llvm.store %[[VAL_0]], %[[VAL_4]] : i32, !llvm.ptr
// CHECK:           %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(struct<(i8)>, struct<(i32, i32)>, i32)>, i32, i8)>
// CHECK:           llvm.store %[[VAL_0]], %[[VAL_5]] : i32, !llvm.ptr
// CHECK:           %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_3]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i8)>, struct<(i32, i32)>, i32)>
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.null : !llvm.ptr
// CHECK:           %[[VAL_8:.*]] = llvm.icmp "ne" %[[VAL_6]], %[[VAL_7]] : !llvm.ptr
// CHECK:           %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_6]], %[[VAL_7]] : !llvm.ptr
// CHECK:           %[[VAL_10:.*]] = call @_Z8castAtoCP1A(%[[VAL_9]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_10]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i8)>, struct<(i32, i32)>, i32)>
// CHECK:           %[[VAL_12:.*]] = llvm.load %[[VAL_11]] : !llvm.ptr -> i32
// CHECK:           %[[VAL_13:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(struct<(i8)>, struct<(i32, i32)>, i32)>, i32, i8)>
// CHECK:           %[[VAL_14:.*]] = llvm.getelementptr inbounds %[[VAL_13]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i8)>, struct<(i32, i32)>, i32)>
// CHECK:           %[[VAL_15:.*]] = llvm.icmp "ne" %[[VAL_14]], %[[VAL_7]] : !llvm.ptr
// CHECK:           %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_14]], %[[VAL_7]] : !llvm.ptr
// CHECK:           %[[VAL_17:.*]] = call @_Z8castBtoDP1B(%[[VAL_16]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           %[[VAL_18:.*]] = llvm.getelementptr inbounds %[[VAL_17]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(struct<(i8)>, struct<(i32, i32)>, i32)>, i32, i8)>
// CHECK:           %[[VAL_19:.*]] = llvm.load %[[VAL_18]] : !llvm.ptr -> i32
// CHECK:           %[[VAL_20:.*]] = arith.addi %[[VAL_12]], %[[VAL_19]] : i32
// CHECK:           %[[VAL_21:.*]] = llvm.getelementptr inbounds %[[VAL_13]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i8)>, struct<(i32, i32)>, i32)>
// CHECK:           %[[VAL_22:.*]] = llvm.icmp "ne" %[[VAL_21]], %[[VAL_7]] : !llvm.ptr
// CHECK:           %[[VAL_23:.*]] = arith.select %[[VAL_22]], %[[VAL_21]], %[[VAL_7]] : !llvm.ptr
// CHECK:           %[[VAL_24:.*]] = call @_Z8castAtoDP1A(%[[VAL_23]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           %[[VAL_25:.*]] = llvm.getelementptr inbounds %[[VAL_24]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(struct<(i8)>, struct<(i32, i32)>, i32)>, i32, i8)>
// CHECK:           %[[VAL_26:.*]] = llvm.load %[[VAL_25]] : !llvm.ptr -> i32
// CHECK:           %[[VAL_27:.*]] = arith.addi %[[VAL_20]], %[[VAL_26]] : i32
// CHECK:           return %[[VAL_27]] : i32
// CHECK:         }
int main() {
    C c;
    D d;
    c.val3 = 2;
    d.val3 = 2;
    return castAtoC(&c)->val3 + // expect nonzero offset due to A -> C
           castBtoD(&d)->val3 +
           castAtoD(&d)->val3; // expect nonzero offset due to A -> C
}
