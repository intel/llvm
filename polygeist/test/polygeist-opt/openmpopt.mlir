// RUN: polygeist-opt --openmp-opt --split-input-file %s | FileCheck %s

module {
  func.func private @inner(index) -> ()
  func.func private @inner2(index, index) -> ()
  func.func @moo(%start : index, %end : index, %step : index, %c : i1) {
    scf.for %arg15 = %start to %end step %step {
      omp.parallel   {
        func.call @inner(%arg15) : (index) -> ()
        omp.terminator
      }
    }
    scf.for %arg15 = %start to %end step %step {
      omp.parallel   {
        scf.for %arg16 = %start to %end step %step {
          func.call @inner2(%arg15, %arg16) : (index, index) -> ()
        }
        omp.terminator
      }
    }
    scf.if %c {
      omp.parallel   {
        func.call @inner(%start) : (index) -> ()
        omp.terminator
      }
    }
    return
  }
}

// CHECK:   func.func @moo(%arg0: index, %arg1: index, %arg2: index, %arg3: i1) {
// CHECK-NEXT:     omp.parallel   {
// CHECK-NEXT:       scf.for %arg4 = %arg0 to %arg1 step %arg2 {
// CHECK-NEXT:         func.call @inner(%arg4) : (index) -> ()
// CHECK-NEXT:         omp.barrier
// CHECK-NEXT:       }
// CHECK-NEXT:       omp.barrier
// CHECK-NEXT:       scf.for %arg4 = %arg0 to %arg1 step %arg2 {
// CHECK-NEXT:         scf.for %arg5 = %arg0 to %arg1 step %arg2 {
// CHECK-NEXT:           func.call @inner2(%arg4, %arg5) : (index, index) -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:         omp.barrier
// CHECK-NEXT:       }
// CHECK-NEXT:       omp.barrier
// CHECK-NEXT:       scf.if %arg3 {
// CHECK-NEXT:         memref.alloca_scope  {
// CHECK-NEXT:           func.call @inner(%arg0) : (index) -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       omp.terminator
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
