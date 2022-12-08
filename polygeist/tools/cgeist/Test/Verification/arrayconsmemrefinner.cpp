// RUN: cgeist -O0 -w %s --function=* -S | FileCheck %s

struct AIntDivider {
    AIntDivider() : divisor(3) {}
    unsigned int divisor;
};

struct Meta {
    AIntDivider sizes_[25];
    double x;
};

void kern() {
    Meta m;
}

// CHECK:   func @_Z4kernv() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<(array<25 x struct<(i32)>>, f64)> : (i64) -> !llvm.ptr<struct<(array<25 x struct<(i32)>>, f64)>>
// CHECK-NEXT:     call @_ZN4MetaC1Ev(%0) : (!llvm.ptr<struct<(array<25 x struct<(i32)>>, f64)>>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN4MetaC1Ev(%arg0: !llvm.ptr<struct<(array<25 x struct<(i32)>>, f64)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-DAG:     %c25 = arith.constant 25 : index
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %0 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr<struct<(array<25 x struct<(i32)>>, f64)>>) -> !llvm.ptr<array<25 x struct<(i32)>>>
// CHECK-NEXT:     %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr<array<25 x struct<(i32)>>>) -> !llvm.ptr<struct<(i32)>>
// CHECK-NEXT:     scf.for %arg1 = %c0 to %c25 step %c1 {
// CHECK-NEXT:       %2 = arith.index_cast %arg1 : index to i64
// CHECK-NEXT:       %3 = llvm.getelementptr %1[%2] : (!llvm.ptr<struct<(i32)>>, i64) -> !llvm.ptr<struct<(i32)>>
// CHECK-NEXT:       call @_ZN11AIntDividerC1Ev(%3) : (!llvm.ptr<struct<(i32)>>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func.func @_ZN11AIntDividerC1Ev(%arg0: !llvm.ptr<struct<(i32)>>)  attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %c3_i32 = arith.constant 3 : i32
// CHECK-NEXT:     %0 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr<struct<(i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %c3_i32, %0 : !llvm.ptr<i32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
