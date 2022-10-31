// RUN: polygeist-opt --cpuify="method=distribute" -allow-unregistered-dialect -canonicalize --split-input-file %s | FileCheck %s

module {
  func.func private @use(%a : i1) -> ()
  func.func private @make() -> (i1)
  func.func private @something() -> ()
  func.func @fast(%arg0: i32, %c : i1, %25 : memref<9x9xi1>, %cond : i1) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %false = arith.constant false
    %c9 = arith.constant 9 : index
    %true = arith.constant true
      %23 = memref.alloca() : memref<256xi32>
      scf.parallel (%arg4) = (%c0) to (%c9) step (%c1) {
          %r = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %false) -> (i1) {
            %s = scf.if %cond -> (i1) {
              %m = arith.xori %arg2, %true : i1
              "polygeist.barrier"(%arg4) : (index) -> ()
              "test.something"() : () -> ()
              scf.yield %m : i1
            } else {
              scf.yield %arg2 : i1
            }
            scf.yield %s : i1
          }
          func.call @use(%r) : (i1) -> () 
          scf.yield
      }
    return
  }
}

// CHECK:   func.func @fast(%arg0: i32, %arg1: i1, %arg2: memref<9x9xi1>, %arg3: i1) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %true = arith.constant true
// CHECK-DAG:     %c9 = arith.constant 9 : index
// CHECK-DAG:     %false = arith.constant false
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c10 = arith.constant 10 : index
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %alloca = memref.alloca() : memref<9xi1>
// CHECK-NEXT:     scf.parallel (%arg4) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:       memref.store %false, %alloca[%arg4] : memref<9xi1>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
//    TODO don't need this cache during parallel split
// CHECK-NEXT:     %alloca_0 = memref.alloca() : memref<9xi1>
// CHECK-NEXT:     scf.for %arg4 = %c0 to %c10 step %c1 {
// CHECK-NEXT:       scf.if %arg3 {
// CHECK-NEXT:         scf.parallel (%arg5) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:           %0 = memref.load %alloca[%arg5] : memref<9xi1>
// CHECK-NEXT:           %1 = arith.xori %0, %true : i1
// CHECK-NEXT:           memref.store %1, %alloca_0[%arg5] : memref<9xi1>
// CHECK-NEXT:           scf.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.parallel (%arg5) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:           %0 = memref.load %alloca_0[%arg5] : memref<9xi1>
// CHECK-NEXT:           "test.something"() : () -> ()
// CHECK-NEXT:           memref.store %0, %alloca[%arg5] : memref<9xi1>
// CHECK-NEXT:           scf.yield
// CHECK-NEXT:         }
// CHECK-NEXT:       } else {
//    TODO don't need load/store
// CHECK-NEXT:         scf.parallel (%arg5) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:           %0 = memref.load %alloca[%arg5] : memref<9xi1>
// CHECK-NEXT:           memref.store %0, %alloca[%arg5] : memref<9xi1>
// CHECK-NEXT:           scf.yield
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.parallel (%arg4) = (%c0) to (%c9) step (%c1) {
// CHECK-NEXT:       %0 = memref.load %alloca[%arg4] : memref<9xi1>
// CHECK-NEXT:       func.call @use(%0) : (i1) -> ()
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
