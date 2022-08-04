// RUN: cgeist %s --function=* -fopenmp -S | FileCheck %s

void square(double* x, int sstart, int send, int sinc) {
    #pragma omp parallel for
    for(int i=sstart; i < send; i+= sinc) {
        x[i] = i;
    }
}

// CHECK:   func @square(%arg0: memref<?xf64>, %arg1: i32, %arg2: i32, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:     %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     %1 = arith.subi %arg2, %arg1 : i32
// CHECK-NEXT:     %2 = arith.addi %1, %c-1_i32 : i32
// CHECK-NEXT:     %3 = arith.addi %2, %arg3 : i32
// CHECK-NEXT:     %4 = arith.divui %3, %arg3 : i32
// CHECK-NEXT:     %5 = arith.muli %4, %arg3 : i32
// CHECK-NEXT:     %6 = arith.addi %arg1, %5 : i32
// CHECK-NEXT:     %7 = arith.index_cast %6 : i32 to index
// CHECK-NEXT:     %8 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:     scf.parallel (%arg4) = (%0) to (%7) step (%8) {
// CHECK-NEXT:       %9 = arith.index_cast %arg4 : index to i32
// CHECK-NEXT:       %10 = arith.sitofp %9 : i32 to f64
// CHECK-NEXT:       memref.store %10, %arg0[%arg4] : memref<?xf64>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
