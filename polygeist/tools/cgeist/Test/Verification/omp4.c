// RUN: cgeist %s --function=* -fopenmp -S --raise-scf-to-affine=false | FileCheck %s

int get(int);
void square(double* x, int ss) {
    int i=7;
    #pragma omp parallel for private(i)
    for(i=get(ss); i < 10; i+= 2) {
        x[i] = i;
        i++;
        x[i] = i;
    }
}

// CHECK:   func @square(%arg0: memref<?xf64>, %arg1: i32)
// CHECK-DAG:     %c11_i32 = arith.constant 11 : i32
// CHECK-DAG:     %c2 = arith.constant 2 : index
// CHECK-DAG:     %c1_i32 = arith.constant 1 : i32
// CHECK-DAG:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:     %0 = call @get(%arg1) : (i32) -> i32
// CHECK-NEXT:     %[[a1:.+]] = arith.index_cast %0 : i32 to index

// CHECK-NEXT:     %2 = arith.subi %c11_i32, %0 : i32
// CHECK-NEXT:     %3 = arith.divui %2, %c2_i32 : i32
// CHECK-NEXT:     %4 = arith.muli %3, %c2_i32 : i32
// CHECK-NEXT:     %5 = arith.addi %0, %4 : i32
// CHECK-NEXT:     %[[a5:.+]] = arith.index_cast %5 : i32 to index

// CHECK-NEXT:     scf.parallel (%arg2) = (%[[a1]]) to (%[[a5]]) step (%c2) {
// CHECK-NEXT:       %[[a6:.+]] = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:       %[[a7:.+]] = arith.sitofp %[[a6]] : i32 to f64
// CHECK-NEXT:       memref.store %[[a7]], %arg0[%arg2] : memref<?xf64>
// CHECK-NEXT:       %[[a8:.+]] = arith.addi %[[a6]], %c1_i32 : i32
// CHECK-NEXT:       %[[a9:.+]] = arith.sitofp %[[a8]] : i32 to f64
// CHECK-NEXT:       %[[a10:.+]] = arith.index_cast %[[a8]] : i32 to index
// CHECK-NEXT:       memref.store %[[a9]], %arg0[%[[a10]]] : memref<?xf64>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
