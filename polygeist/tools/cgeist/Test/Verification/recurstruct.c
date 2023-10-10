// RUN: cgeist %s --function=sum -S | FileCheck %s

struct Node {
    struct Node* next;
    double value;
};

double sum(struct Node* n) {
    if (n == 0) return 0;
    return n->value + sum(n->next);
}

// CHECK-LABEL:   func.func @sum(
// CHECK-SAME:                   %[[VAL_0:.*]]: !llvm.ptr) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.icmp "eq" %[[VAL_0]], %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      %[[VAL_4:.*]] = scf.if %[[VAL_3]] -> (f64) {
// CHECK-NEXT:        scf.yield %[[VAL_1]] : f64
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"polygeist@mlir@struct.Node", (ptr, f64)>
// CHECK-NEXT:        %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> f64
// CHECK-NEXT:        %[[VAL_7:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"polygeist@mlir@struct.Node", (ptr, f64)>
// CHECK-NEXT:        %[[VAL_8:.*]] = llvm.load %[[VAL_7]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:        %[[VAL_9:.*]] = func.call @sum(%[[VAL_8]]) : (!llvm.ptr) -> f64
// CHECK-NEXT:        %[[VAL_10:.*]] = arith.addf %[[VAL_6]], %[[VAL_9]] : f64
// CHECK-NEXT:        scf.yield %[[VAL_10]] : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      return %[[VAL_11:.*]] : f64
// CHECK-NEXT:    }
