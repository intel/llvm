// RUN: cgeist -O0 -w  %s --function=* -S --raise-scf-to-affine=false | FileCheck %s

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

// CHECK-LABEL:   func.func @_Z4kernv() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(array<25 x struct<(i32)>>, f64)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      call @_ZN4MetaC1Ev(%[[VAL_1]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZN4MetaC1Ev(
// CHECK-SAME:                            %[[VAL_0:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 25 : index
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<25 x struct<(i32)>>, f64)>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_4]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<25 x struct<(i32)>>
// CHECK-NEXT:      scf.for %[[VAL_6:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_1]] {
// CHECK-NEXT:        %[[VAL_7:.*]] = arith.index_cast %[[VAL_6]] : index to i64
// CHECK-NEXT:        %[[VAL_8:.*]] = llvm.getelementptr %[[VAL_5]]{{\[}}%[[VAL_7]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32)>
// CHECK-NEXT:        func.call @_ZN11AIntDividerC1Ev(%[[VAL_8]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZN11AIntDividerC1Ev(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 3 : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32)>
// CHECK-NEXT:      llvm.store %[[VAL_1]], %[[VAL_2]] : i32, !llvm.ptr
// CHECK-NEXT:      return
// CHECK-NEXT:    }
