// RUN: cgeist %s --function=* -S | FileCheck %s

void* calloc(unsigned long a, unsigned long b);

float* zmem(int n) {
    float* out = (float*)calloc(sizeof(float), n);
    return out;
}

// CHECK:   func @zmem(%arg0: i32) -> memref<?xf32> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %c4 = arith.constant 4 : index
// CHECK-DAG:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:     %1 = arith.muli %0, %c4 : index
// CHECK-NEXT:     %2 = arith.divui %1, %c4 : index
// CHECK-NEXT:     %alloc = memref.alloc(%2) : memref<?xf32>
// CHECK-NEXT:     scf.for %arg1 = %c0 to %2 step %c1 {
// CHECK-NEXT:       memref.store %cst, %alloc[%arg1] : memref<?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %alloc : memref<?xf32>
// CHECK-NEXT:   }
