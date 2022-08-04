// RUN: cgeist %s --function=sum -S | FileCheck %s

struct Node {
    struct Node* next;
    double value;
};

double sum(struct Node* n) {
    if (n == 0) return 0;
    return n->value + sum(n->next);
}

// CHECK:   func.func @sum(%arg0: !llvm.ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-DAG:     %cst = arith.constant 0.000000e+00 : f64
// CHECK-DAG:     %0 = llvm.mlir.null : !llvm.ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>
// CHECK-NEXT:     %1 = llvm.icmp "eq" %arg0, %0 : !llvm.ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>
// CHECK-NEXT:     %2 = scf.if %1 -> (f64) {
// CHECK-NEXT:       scf.yield %cst : f64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       %3 = llvm.getelementptr %arg0[%c0_i32, 1] : (!llvm.ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>, i32) -> !llvm.ptr<f64>
// CHECK-NEXT:       %4 = llvm.load %3 : !llvm.ptr<f64>
// CHECK-NEXT:       %5 = llvm.getelementptr %arg0[%c0_i32, 0] : (!llvm.ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>, i32) -> !llvm.ptr<ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>>
// CHECK-NEXT:       %6 = llvm.load %5 : !llvm.ptr<ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>>
// CHECK-NEXT:       %7 = func.call @sum(%6) : (!llvm.ptr<struct<"polygeist@mlir@struct.Node", (ptr<struct<"polygeist@mlir@struct.Node">>, f64)>>) -> f64
// CHECK-NEXT:       %8 = arith.addf %4, %7 : f64
// CHECK-NEXT:       scf.yield %8 : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     return %2 : f64
// CHECK-NEXT:   }
