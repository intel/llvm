// RUN: cgeist %s -detect-reduction --function=kernel_nussinov -S | FileCheck %s
// RUN: cgeist %s -detect-reduction --function=kernel_nussinov -S -memref-fullrank | FileCheck %s --check-prefix=FULLRANK

#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

void set(double table[20]);

void kernel_nussinov(double* out, int n)  {
  double table[20];
  set(table);
#pragma scop
   for (int k=0; k<10; k++) {
      out[n] = max_score(out[n], table[k]);
   }
#pragma endscop
}

// CHECK:   func @kernel_nussinov(%arg0: memref<?xf64>, %arg1: i32)
// CHECK-DAG:     %[[i0:.+]] = memref.alloca() : memref<20xf64>
// CHECK-DAG:     %[[i1:.+]] = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     %cast = memref.cast %[[i0]] : memref<20xf64> to memref<?xf64>
// CHECK-NEXT:     call @set(%cast) : (memref<?xf64>) -> ()
// CHECK-NEXT:     %1 = affine.load %arg0[symbol(%[[i1]])] : memref<?xf64>
// CHECK-NEXT:     %2 = affine.for %arg2 = 0 to 10 iter_args(%arg3 = %1) -> (f64) {
// CHECK-NEXT:       %3 = affine.load %[[i0]][%arg2] : memref<20xf64>
// CHECK-NEXT:       %4 = arith.cmpf uge, %arg3, %3 : f64
// CHECK-NEXT:       %5 = arith.select %4, %arg3, %3 : f64
// CHECK-NEXT:       affine.yield %5 : f64
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.store %2, %arg0[symbol(%[[i1]])] : memref<?xf64>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// FULLRANK: func @kernel_nussinov(%{{.*}}: memref<?xf64>, %{{.*}}: i32)
// FULLRANK:     %[[i0:.+]] = memref.alloca() : memref<20xf64>
// FULLRANK:     call @set(%[[i0]]) : (memref<20xf64>) -> ()
