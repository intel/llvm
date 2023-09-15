// RUN: cgeist %s --function=* -fopenmp -S | FileCheck %s

void square2(double** x, int sstart, int send, int sinc, int tstart, int tend, int tinc) {
    #pragma omp parallel for collapse(2)
    for(int i=sstart; i < send; i+= sinc) {
    for(int j=tstart; j < tend; j+= tinc) {
        x[i][j] = i + j;
    }
    }
}


// CHECK:   func @square2(%arg0: memref<?xmemref<?xf64>>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:     %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     %1 = arith.index_cast %arg4 : i32 to index
// CHECK-NEXT:     %2 = arith.subi %arg2, %arg1 : i32
// CHECK-NEXT:     %3 = arith.addi %2, %c-1_i32 : i32
// CHECK-NEXT:     %4 = arith.addi %3, %arg3 : i32
// CHECK-NEXT:     %5 = arith.divui %4, %arg3 : i32
// CHECK-NEXT:     %6 = arith.muli %5, %arg3 : i32
// CHECK-NEXT:     %7 = arith.addi %arg1, %6 : i32
// CHECK-NEXT:     %8 = arith.index_cast %7 : i32 to index
// CHECK-NEXT:     %9 = arith.subi %arg5, %arg4 : i32
// CHECK-NEXT:     %10 = arith.addi %9, %c-1_i32 : i32
// CHECK-NEXT:     %11 = arith.addi %10, %arg6 : i32
// CHECK-NEXT:     %12 = arith.divui %11, %arg6 : i32
// CHECK-NEXT:     %13 = arith.muli %12, %arg6 : i32
// CHECK-NEXT:     %14 = arith.addi %arg4, %13 : i32
// CHECK-NEXT:     %15 = arith.index_cast %14 : i32 to index
// CHECK-NEXT:     %16 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:     %17 = arith.index_cast %arg6 : i32 to index
// CHECK-NEXT:     scf.parallel (%arg7, %arg8) = (%0, %1) to (%8, %15) step (%16, %17) {
// CHECK-NEXT:       %18 = arith.index_cast %arg7 : index to i64
// CHECK-NEXT:       %19 = arith.index_cast %arg8 : index to i64
// CHECK-NEXT:       %20 = arith.addi %18, %19 : i64
// CHECK-NEXT:       %21 = arith.sitofp %20 : i64 to f64
// CHECK-NEXT:       %22 = memref.load %arg0[%arg7] : memref<?xmemref<?xf64>>
// CHECK-NEXT:       memref.store %21, %22[%arg8] : memref<?xf64>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
