// RUN: cgeist %s -O2 --function=* -S | FileCheck %s

void histo_kernel(int i);

int runHisto(int totalNum, int partialNum) {
    for(int i = 0; i < totalNum; i+=partialNum*2)
    {
        histo_kernel(i);
    }
    return 0;
}

// CHECK:   func @runHisto(%arg0: i32, %arg1: i32) -> i32
// CHECK-DAG:     %c2_i32 = arith.constant 2 : i32
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %0 = arith.muli %arg1, %c2_i32 : i32
// CHECK-NEXT:     %1 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:     %2 = arith.index_cast %0 : i32 to index
// CHECK-NEXT:     scf.for %arg2 = %c0 to %1 step %2 {
// CHECK-NEXT:       %3 = arith.divui %arg2, %2 : index
// CHECK-NEXT:       %4 = arith.muli %3, %2 : index
// CHECK-NEXT:       %5 = arith.index_cast %4 : index to i32
// CHECK-NEXT:       call @histo_kernel(%5) : (i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
