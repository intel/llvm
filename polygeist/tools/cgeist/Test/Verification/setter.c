// RUN: cgeist %s --function=* -S | FileCheck %s

void sub0(int *a);
void sub(int *a) {
    *a = 3;
}

void kernel_deriche() {
    int a;
    sub0(&a);
}

// CHECK:  func @sub(%arg0: memref<?xi32>)
// CHECK-NEXT:    %c3_i32 = arith.constant 3 : i32
// CHECK-NEXT:    affine.store %c3_i32, %arg0[0] : memref<?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK:  func @kernel_deriche()
// CHECK-NEXT:    %alloca = memref.alloca() : memref<1xi32>
// CHECK-NEXT:    %0 = llvm.mlir.undef : i32
// CHECK-NEXT:    affine.store %0, %alloca[0] : memref<1xi32>
// CHECK-NEXT:    %cast = memref.cast %alloca : memref<1xi32> to memref<?xi32>
// CHECK-NEXT:    call @sub0(%cast) : (memref<?xi32>) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }

