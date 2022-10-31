// RUN: cgeist %s --function=* -S | FileCheck %s

extern "C" {

void sub0(int& a);
void sub(int& a) {
    a++;
}

void kernel_deriche() {
    int a = 32;;
    int &b = a;
    sub0(b);
}

}

// CHECK:   func @sub(%arg0: memref<?xi32>)
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %0 = affine.load %arg0[0] : memref<?xi32>
// CHECK-NEXT:     %1 = arith.addi %0, %c1_i32 : i32
// CHECK-NEXT:     affine.store %1, %arg0[0] : memref<?xi32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @kernel_deriche()
// CHECK-NEXT:     %c32_i32 = arith.constant 32 : i32
// CHECK-NEXT:     %alloca = memref.alloca() : memref<1xi32>
// CHECK-NEXT:     %cast = memref.cast %alloca : memref<1xi32> to memref<?xi32>
// CHECK-NEXT:     affine.store %c32_i32, %alloca[0] : memref<1xi32>
// CHECK-NEXT:     call @sub0(%cast) : (memref<?xi32>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
