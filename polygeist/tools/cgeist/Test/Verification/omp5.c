// RUN: cgeist %s --function=* -fopenmp -S --raise-scf-to-affine=false | FileCheck %s

void square(double* x, int sstart, int send, int sinc) {
    #pragma omp parallel for
    for(int i=sstart; i < send; i++) {
        x[i] = i;
    }
}

// CHECK:   func @square(%arg0: memref<?xf64>, %arg1: i32, %arg2: i32, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %[[i0:.+]] = arith.index_cast %arg1 : i32 to index
// CHECK-DAG:     %[[i1:.+]] = arith.index_cast %arg2 : i32 to index
// CHECK-NEXT:     scf.parallel (%arg4) = (%[[i0]]) to (%[[i1]]) step (%c1) {
// CHECK-NEXT:       %2 = arith.index_cast %arg4 : index to i32
// CHECK-NEXT:       %3 = arith.sitofp %2 : i32 to f64
// CHECK-NEXT:       memref.store %3, %arg0[%arg4] : memref<?xf64>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
