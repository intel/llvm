// RUN: cgeist %s -O2 --function=kernel_correlation --raise-scf-to-affine -S | FileCheck %s

#define DATA_TYPE double

#define SCALAR_VAL(x) ((double)x)

void use(int i);

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_correlation(int start, int end) {
  for (int i = end; i >= start; i--) {
    use(i);
  }
}

// CHECK: #map = affine_map<()[s0] -> (s0 + 1)>
// CHECK: kernel_correlation
// CHECK-DAG:     %c-1 = arith.constant -1 : index
// CHECK-NEXT:     %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     %1 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:     affine.for %arg2 = %1 to #map()[%0] {
// CHECK-NEXT:       %2 = arith.subi %arg2, %1 : index
// CHECK-NEXT:       %3 = arith.muli %2, %c-1 : index
// CHECK-NEXT:       %4 = arith.addi %0, %3 : index
// CHECK-NEXT:       %5 = arith.index_cast %4 : index to i32
// CHECK-NEXT:       call @use(%5) : (i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
