// RUN: cgeist  %s --function=* -c -S | FileCheck %s

float *foo(float *a) {
	a += 32;
	return a;
}
// CHECK: func @_Z3fooPf(%arg0: memref<?xf32>)
// CHECK-NEXT   %c32 = arith.constant 32 : index
// CHECK-NEXT   %0 = "polygeist.subindex"(%arg0, %c32) : (memref<?xf32>, index) -> memref<?xf32>
// CHECK-NEXT   return %0 : memref<?xf32>
// CHECK-NEXT }

