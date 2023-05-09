// RUN: polygeist-opt --polygeist-mem2reg --split-input-file %s | FileCheck %s

module  {
  llvm.func @print(i32)
  func.func @h(%arg7: i1, %arg8: i1, %arg9 : i1) {
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %2 = memref.alloca() : memref<i32>
    %3 = llvm.mlir.undef : i32
    memref.store %3, %2[] : memref<i32>
    scf.if %arg8 {
      memref.store %c-1_i32, %2[] : memref<i32>
      scf.execute_region {
        memref.store %c0_i32, %2[] : memref<i32>
        scf.yield
      }
      scf.if %arg9 {
          memref.store %c5_i32, %2[] : memref<i32>
      }
      %23 = memref.load %2[] : memref<i32>
      llvm.call @print(%23) : (i32) -> ()
    }
    return
  }
}

// CHECK:   func.func @h(%arg0: i1, %arg1: i1, %arg2: i1)
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:     %0 = llvm.mlir.undef : i32
// CHECK-NEXT:     scf.if %arg1 {
// CHECK-NEXT:       scf.execute_region {
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       %1 = scf.if %arg2 -> (i32) {
// CHECK-NEXT:         scf.yield %c5_i32 : i32
// CHECK-NEXT:       } else {
// CHECK-NEXT:         scf.yield %c0_i32 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       llvm.call @print(%1) : (i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

