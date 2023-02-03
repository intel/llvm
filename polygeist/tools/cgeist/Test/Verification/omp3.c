// RUN: cgeist %s --function=* -fopenmp -S --raise-scf-to-affine=false | FileCheck %s

void square(double* x) {
    int i;
    #pragma omp parallel for private(i)
    for(i=3; i < 10; i+= 2) {
        x[i] = i;
        i++;
        x[i] = i;
    }
}

// CHECK:   func @square(%arg0: memref<?xf64>)
// CHECK-DAG:     %c2 = arith.constant 2 : index
// CHECK-DAG:     %c11 = arith.constant 11 : index
// CHECK-DAG:     %c1_i32 = arith.constant 1 : i32
// CHECK-DAG:     %c3 = arith.constant 3 : index
// CHECK-NEXT:     scf.parallel (%arg1) = (%c3) to (%c11) step (%c2) {
// CHECK-NEXT:       %0 = arith.index_cast %arg1 : index to i32
// CHECK-NEXT:       %1 = arith.sitofp %0 : i32 to f64
// CHECK-NEXT:       memref.store %1, %arg0[%arg1] : memref<?xf64>
// CHECK-NEXT:       %2 = arith.addi %0, %c1_i32 : i32
// CHECK-NEXT:       %3 = arith.sitofp %2 : i32 to f64
// CHECK-NEXT:       %4 = arith.index_cast %2 : i32 to index
// CHECK-NEXT:       memref.store %3, %arg0[%4] : memref<?xf64>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
