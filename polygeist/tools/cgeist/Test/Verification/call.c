// RUN: cgeist %s --function=* -S | FileCheck %s
// RUN: cgeist %s --function=* -S -memref-fullrank | FileCheck %s --check-prefix=FULLRANK

void sub0(int a[2]);
void sub(int a[2]) { a[2]++; }

void kernel_deriche() {
  int a[2];
  sub0(a);
}

// FULLRANK:   @sub(%arg0: memref<2xi32>)
// CHECK:   @sub(%arg0: memref<?xi32>)
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %0 = affine.load %arg0[2] : memref<?xi32>
// CHECK-NEXT:    %1 = arith.addi %0, %c1_i32 : i32
// CHECK-NEXT:    affine.store %1, %arg0[2] : memref<?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK:  func @kernel_deriche()
// CHECK-NEXT:    %alloca = memref.alloca() : memref<2xi32>
// CHECK-NEXT:    %cast = memref.cast %alloca : memref<2xi32> to memref<?xi32>
// CHECK-NEXT:    call @sub0(%cast) : (memref<?xi32>) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
