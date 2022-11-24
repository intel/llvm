// RUN: cgeist %s -O0 --function=* -S | FileCheck %s

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

// CHECK:   func @_Z4oaddRi(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %0 = affine.load %arg0[0] : memref<?xi32>
// CHECK-NEXT:     %1 = arith.addi %0, %c1_i32 : i32
// CHECK-NEXT:     affine.store %1, %arg0[0] : memref<?xi32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_Z1QR1A(%arg0: memref<?x1xmemref<?xi32>>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     call @_ZN1A3addEv(%arg0) : (memref<?x1xmemref<?xi32>>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN1A3addEv(%arg0: memref<?x1xmemref<?xi32>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %0 = affine.load %arg0[0, 0] : memref<?x1xmemref<?xi32>>
// CHECK-NEXT:     call @_Z4oaddRi(%0) : (memref<?xi32>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
