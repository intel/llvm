// RUN: polygeist-opt --mem2reg --split-input-file %s | FileCheck %s

module {
  func.func private @overwrite(%a : memref<i1>)
  func.func private @use(%a : i1)
  
  func.func @infLoop1(%c : i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %false = arith.constant false
    %6 = memref.alloca() : memref<i1>
    memref.store %true, %6[] : memref<i1>
        scf.execute_region {
          cf.br ^bb1(%c0_i32 : i32)
        ^bb1(%28 : i32):  // 2 preds: ^bb0, ^bb2
          %29 = arith.cmpi slt, %28, %c2_i32 : i32
          %30 = memref.load %6[] : memref<i1>
          %33 = arith.andi %29, %30 : i1
          cf.cond_br %33, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          scf.if %c {
              %45 = arith.cmpi eq, %28, %c1_i32 : i32
              scf.if %45 {
                memref.store %false, %6[] : memref<i1>
              }
          }
              %44 = memref.load %6[] : memref<i1>
          func.call @use(%44) : (i1) -> ()
          %42 = arith.addi %28, %c1_i32 : i32
          cf.br ^bb1(%42 : i32)
        ^bb3:  // pred: ^bb1
          scf.yield
        }
    return
  }

// CHECK:   func.func @infLoop1
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:     %true = arith.constant true
// CHECK-NEXT:     %false = arith.constant false
// CHECK-NEXT:     scf.execute_region {
// CHECK-NEXT:       cf.br ^bb1(%c0_i32, %true : i32, i1)
// CHECK-NEXT:     ^bb1(%0: i32, %1: i1):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:       %2 = arith.cmpi slt, %0, %c2_i32 : i32
// CHECK-NEXT:       %3 = arith.andi %2, %1 : i1
// CHECK-NEXT:       cf.cond_br %3, ^bb2, ^bb3
// CHECK-NEXT:     ^bb2:  // pred: ^bb1
// CHECK-NEXT:       %4 = scf.if %arg0 -> (i1) {
// CHECK-NEXT:         %6 = arith.cmpi eq, %0, %c1_i32 : i32
// CHECK-NEXT:         %7 = scf.if %6 -> (i1) {
// CHECK-NEXT:           scf.yield %false : i1
// CHECK-NEXT:         } else {
// CHECK-NEXT:           scf.yield %1 : i1
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %7 : i1
// CHECK-NEXT:       } else {
// CHECK-NEXT:         scf.yield %1 : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       func.call @use(%4) : (i1) -> ()
// CHECK-NEXT:       %5 = arith.addi %0, %c1_i32 : i32
// CHECK-NEXT:       cf.br ^bb1(%5, %4 : i32, i1)
// CHECK-NEXT:     ^bb3:  // pred: ^bb1
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

  func.func @infLoop2(%c : i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %false = arith.constant false
    %6 = memref.alloca() : memref<i1>
    memref.store %true, %6[] : memref<i1>
        scf.execute_region {
          cf.br ^bb1(%c0_i32 : i32)
        ^bb1(%28 : i32):  // 2 preds: ^bb0, ^bb2
          %29 = arith.cmpi slt, %28, %c2_i32 : i32
          %30 = memref.load %6[] : memref<i1>
          %33 = arith.andi %29, %30 : i1
          cf.cond_br %33, ^bb2, ^bb3
        ^bb2:  // pred: ^bb1
          scf.if %true {
              %45 = arith.cmpi eq, %28, %c1_i32 : i32
              scf.if %45 {
                func.call @overwrite(%6) : (memref<i1>) -> ()
              }
          }
          %44 = memref.load %6[] : memref<i1>
          func.call @use(%44) : (i1) -> ()
          %42 = arith.addi %28, %c1_i32 : i32
          cf.br ^bb1(%42 : i32)
        ^bb3:  // pred: ^bb1
          scf.yield
        }
    return
  }

// CHECK:   func.func @infLoop2
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:     %true = arith.constant true
// CHECK-NEXT:     %false = arith.constant false
// CHECK-NEXT:     %alloca = memref.alloca() : memref<i1>
// CHECK-NEXT:     memref.store %true, %alloca[] : memref<i1>
// CHECK-NEXT:     scf.execute_region {
// CHECK-NEXT:       cf.br ^bb1(%c0_i32 : i32)
// CHECK-NEXT:     ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:       %1 = arith.cmpi slt, %0, %c2_i32 : i32
// CHECK-NEXT:       %2 = memref.load %alloca[] : memref<i1>
// CHECK-NEXT:       %3 = arith.andi %1, %2 : i1
// CHECK-NEXT:       cf.cond_br %3, ^bb2, ^bb3
// CHECK-NEXT:     ^bb2:  // pred: ^bb1
// CHECK-NEXT:       scf.if %true {
// CHECK-NEXT:         %6 = arith.cmpi eq, %0, %c1_i32 : i32
// CHECK-NEXT:         scf.if %6 {
// CHECK-NEXT:           func.call @overwrite(%alloca) : (memref<i1>) -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       %4 = memref.load %alloca[] : memref<i1>
// CHECK-NEXT:       func.call @use(%4) : (i1) -> ()
// CHECK-NEXT:       %5 = arith.addi %0, %c1_i32 : i32
// CHECK-NEXT:       cf.br ^bb1(%5 : i32)
// CHECK-NEXT:     ^bb3:  // pred: ^bb1
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

  func.func @bpnn_train_cuda(%arg0: memref<11xf32>, %arg1: i1) {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c10 = arith.constant 10 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %1 = llvm.mlir.undef : i32
    %2 = memref.alloca() : memref<i32>
    memref.store %1, %2[] : memref<i32>
    cf.br ^bb3
  ^bb3:  // pred: ^bb1
    scf.if %arg1 {
      memref.store %c1_i32, %2[] : memref<i32>
      scf.execute_region {
        cf.br ^bb1(%c0 : index)
      ^bb1(%11 : index):  // 2 preds: ^bb0, ^bb2
        %12 = arith.cmpi slt, %11, %c10 : index
        cf.cond_br %12, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        %14 = memref.load %arg0[%11] : memref<11xf32>
        %15 = arith.cmpf ugt, %14, %cst : f32
        scf.if %15 {
          memref.store %c0_i32, %2[] : memref<i32>
        }
        %16 = arith.addi %11, %c1 : index
        cf.br ^bb1(%16 : index)
      ^bb3:  // pred: ^bb1
        scf.yield
      }
      %9 = memref.load %2[] : memref<i32>
      func.call @put(%9) : (i32) -> ()
    }
    return
  }
  func.func private @put(%a : i32)
  
  func.func private @_Z12findIndexBinPdiid(%arg0: i1, %arg1: i32, %arg2 : i1) -> i32 {
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = memref.alloca() : memref<i32>
    scf.execute_region {
      memref.store %arg1, %0[] : memref<i32>
      scf.if %arg0 {
        scf.execute_region {
          memref.store %c2_i32, %0[] : memref<i32>
          scf.yield
        }
        scf.if %arg2 {
          %2 = memref.load %0[] : memref<i32>
          %3 = arith.addi %2, %c1_i32 : i32
          memref.store %c1_i32, %0[] : memref<i32>
        }
      }
      scf.yield
    }
    %1 = memref.load %0[] : memref<i32>
    return %1 : i32
  }
}

