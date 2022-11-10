// RUN: cgeist %s %stdinclude -S | FileCheck %s

float A[64][32];

int main() {
#pragma scop
  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 32; j++)
      A[i][j] = 3.0;
#pragma endscop
  return 0;
}

// CHECK:  memref.global @A : memref<64x32xf32> = uninitialized {alignment = 16 : i64}
// CHECK:  func @main() -> i32
// CHECK-DAG:    %cst = arith.constant 3.000000e+00 : f32
// CHECK-DAG:    %c0_i32 = arith.constant 0 : i32
// CHECK-DAG:    %0 = memref.get_global @A : memref<64x32xf32>
// CHECK-NEXT:    affine.for %arg0 = 0 to 64 {
// CHECK-NEXT:      affine.for %arg1 = 0 to 32 {
// CHECK-NEXT:        affine.store %cst, %0[%arg0, %arg1] : memref<64x32xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return %c0_i32 : i32
// CHECK-NEXT:  }
