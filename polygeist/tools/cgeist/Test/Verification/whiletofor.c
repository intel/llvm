// RUN: cgeist %s -O2 --function=whiletofor -S | FileCheck %s
// RUN: cgeist %s -O2 --function=whiletofor -S --memref-fullrank | FileCheck %s --check-prefix=FULLRANK

void use(int a[100][100]);

void whiletofor() {
  int a[100][100];
  int t = 7;
  int i, j;

  for (i = 0; i < 100; i++)
    for (j = 0; j < 100; j++) {
      if (t % 20 == 0)
        a[i][j] = 2;
      else
        a[i][j] = 3;
      t++;
    }

  use(a);
}

// TODO redundant for elim
// CHECK: func @whiletofor()
// CHECK-DAG:     %c7_i32 = arith.constant 7 : i32
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-DAG:     %c20_i32 = arith.constant 20 : i32
// CHECK-DAG:     %c2_i32 = arith.constant 2 : i32
// CHECK-DAG:     %c3_i32 = arith.constant 3 : i32
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-DAG:     %c100 = arith.constant 100 : index
// CHECK-NEXT:     %alloca = memref.alloca() : memref<100x100xi32>
// CHECK-NEXT:     %0 = scf.for %arg0 = %c0 to %c100 step %c1 iter_args(%arg1 = %c7_i32) -> (i32) {
// CHECK-NEXT:       %1 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:       %2 = arith.addi %1, %c100 : index
// CHECK-NEXT:       %3 = arith.index_cast %2 : index to i32
// CHECK-NEXT:       scf.for %arg2 = %c0 to %c100 step %c1 {
// CHECK-NEXT:         %4 = arith.addi %1, %arg2 : index
// CHECK-NEXT:         %5 = arith.index_cast %4 : index to i32
// CHECK-NEXT:         %6 = arith.remsi %5, %c20_i32 : i32
// CHECK-NEXT:         %7 = arith.cmpi eq, %6, %c0_i32 : i32
// CHECK-NEXT:         scf.if %7 {
// CHECK-NEXT:           memref.store %c2_i32, %alloca[%arg0, %arg2] : memref<100x100xi32>
// CHECK-NEXT:         } else {
// CHECK-NEXT:           memref.store %c3_i32, %alloca[%arg0, %arg2] : memref<100x100xi32>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %3 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:      %[[k2:.+]] = memref.cast %alloca : memref<100x100xi32> to memref<?x100xi32>
// CHECK-NEXT:      call @use(%[[k2]]) : (memref<?x100xi32>) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// FULLRANK:      %[[VAL0:.*]] = memref.alloca() : memref<100x100xi32>
// FULLRANK:      call @use(%[[VAL0]]) : (memref<100x100xi32>) -> ()
