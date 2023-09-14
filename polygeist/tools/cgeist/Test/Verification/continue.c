// RUN: cgeist %s --function=* -S --raise-scf-to-affine=false | FileCheck %s

int get();
void other();

int checkCmdLineFlag(const int argc) {
  int bFound = 0;

    for (int i = 1; i < argc; i++) {
      if (get()) {
        bFound = 1;
        continue;
      }
      other();
    }

  return bFound;
}

// CHECK:   func.func @checkCmdLineFlag(%arg0: i32) -> i32
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c1_i32 = arith.constant 1 : i32
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:     %1 = scf.for %arg1 = %c1 to %0 step %c1 iter_args(%arg2 = %c0_i32) -> (i32) {
// CHECK-NEXT:       %2 = func.call @get() : () -> i32
// CHECK-NEXT:       %3 = arith.cmpi ne, %2, %c0_i32 : i32
// CHECK-NEXT:       %4 = arith.select %3, %c1_i32, %arg2 : i32
// CHECK-NEXT:       %5 = arith.cmpi eq, %2, %c0_i32 : i32
// CHECK-NEXT:       scf.if %5 {
// CHECK-NEXT:         call @other() : () -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %4 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %1 : i32
// CHECK-NEXT:   }
