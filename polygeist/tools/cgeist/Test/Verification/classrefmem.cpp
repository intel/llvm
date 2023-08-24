// RUN: cgeist  %s -O0 -w --function=* -S | FileCheck %s

extern int& moo;
void oadd(int& x) {
    x++;
}
struct A {
    void add() {
        oadd(x);
    }
    int &x;
    // TODO int y;
};

void Q(A& a) {
    a.add();
}

// CHECK-LABEL:   func.func @_Z4oaddRi(
// CHECK-SAME:                         %[[VAL_0:.*]]: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = affine.load %[[VAL_0]][0] : memref<?xi32>
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.addi %[[VAL_2]], %[[VAL_1]] : i32
// CHECK-NEXT:      affine.store %[[VAL_3]], %[[VAL_0]][0] : memref<?xi32>
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_Z1QR1A(
// CHECK-SAME:                       %[[VAL_0:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      call @_ZN1A3addEv(%[[VAL_0]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZN1A3addEv(
// CHECK-SAME:                           %[[VAL_0:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi32>)>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> memref<?xi32>
// CHECK-NEXT:      call @_Z4oaddRi(%[[VAL_2]]) : (memref<?xi32>) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }
