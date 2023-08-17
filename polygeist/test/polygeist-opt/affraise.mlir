// RUN: polygeist-opt --raise-scf-to-affine -allow-unregistered-dialect --split-input-file %s | FileCheck %s

module {
  func.func @withinif(%arg0: memref<?xf64>, %arg1: i32, %arg2: memref<?xf64>, %arg3: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.if %arg3 {
      %3 = arith.index_cast %arg1 : i32 to index
      scf.for %arg6 = %c1 to %3 step %c1 {
        %4 = memref.load %arg0[%arg6] : memref<?xf64>
        memref.store %4, %arg2[%arg6] : memref<?xf64>
      }
    }
    return
  }
  func.func @aff(%c : i1, %arg0: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.if %c {
      %75 = arith.index_cast %arg0 : i32 to index
      scf.parallel (%arg5) = (%c0) to (%75) step (%c1) {
        "test.op"() : () -> ()
        scf.yield
      }
    }
    return 
  }
  func.func @complete(%lb: index, %ub: index, %step: index) {
    scf.for %i = %lb to %ub step %step {
      "foo"(%i) : (index) -> ()
      scf.yield
    }
    return
  }
}

// CHECK:   func.func @withinif(%arg0: memref<?xf64>, %arg1: i32, %arg2: memref<?xf64>, %arg3: i1) {
// CHECK-DAG:     %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     scf.if %arg3 {
// CHECK-NEXT:       affine.for %arg4 = 1 to %0 {
// CHECK-NEXT:         %1 = memref.load %arg0[%arg4] : memref<?xf64>
// CHECK-NEXT:         memref.store %1, %arg2[%arg4] : memref<?xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-NEXT:   func.func @aff(%arg0: i1, %arg1: i32) {
// CHECK-NEXT:     %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     scf.if %arg0 {
// CHECK-NEXT:       affine.parallel (%arg2) = (0) to (symbol(%0)) {
// CHECK-NEXT:         "test.op"() : () -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @complete(
// CHECK-SAME:                        %[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index) {
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.subi %[[VAL_1]], %[[VAL_0]] : index
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.ceildivui %[[VAL_3]], %[[VAL_2]] : index
// CHECK-NEXT:      affine.for %[[VAL_5:.*]] = 0 to %[[VAL_4]] {
// CHECK-NEXT:        %[[VAL_6:.*]] = arith.muli %[[VAL_5]], %[[VAL_2]] : index
// CHECK-NEXT:        %[[VAL_7:.*]] = arith.addi %[[VAL_0]], %[[VAL_6]] : index
// CHECK-NEXT:        "foo"(%[[VAL_7]]) : (index) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
