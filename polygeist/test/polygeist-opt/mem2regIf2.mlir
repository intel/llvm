// RUN: polygeist-opt --polygeist-mem2reg --split-input-file %s | FileCheck %s

module {
  func.func @_Z26__device_stub__hotspotOpt1PfS_S_fiiifffffff(%arg0: f32, %arg1 : i1, %arg2 : i1, %arg3 : f32) -> f32 {
    %0 = memref.alloca() : memref<f32>
    %1 = llvm.mlir.undef : f32
    memref.store %1, %0[] : memref<f32>
    %2 = memref.alloca() : memref<f32>
    memref.store %arg3, %2[] : memref<f32>
    %4 = scf.if %arg1 -> (f32) {
      memref.store %1, %2[] : memref<f32>
      memref.store %arg0, %0[] : memref<f32>
      scf.yield %arg0 : f32
    } else {
      scf.yield %1 : f32
    }
    scf.if %arg2 {
      %6 = memref.load %0[] : memref<f32>
      memref.store %4, %2[] : memref<f32>
    }
    %5 = memref.load %2[] : memref<f32>
    return %5 : f32
  }
}

// CHECK:   func.func @_Z26__device_stub__hotspotOpt1PfS_S_fiiifffffff(%arg0: f32, %arg1: i1, %arg2: i1, %arg3: f32) -> f32 {
// CHECK-NEXT:     %0 = llvm.mlir.undef : f32
// CHECK-NEXT:     %1:2 = scf.if %arg1 -> (f32, f32) {
// CHECK-NEXT:       scf.yield %arg0, %0 : f32, f32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %0, %arg3 : f32, f32
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = scf.if %arg2 -> (f32) {
// CHECK-NEXT:       scf.yield %1#0 : f32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %1#1 : f32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %2 : f32
// CHECK-NEXT:   }

// ----

module {
  func.func private @gen() -> (!llvm.ptr)

func.func @_Z3runiPPc(%arg2: i1) -> !llvm.ptr {
  %c1_i64 = arith.constant 1 : i64
  %0 = llvm.alloca %c1_i64 x !llvm.ptr : (i64) -> !llvm.ptr
  %2 = llvm.mlir.zero : !llvm.ptr
  scf.if %arg2 {
    %5 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
    %6 = llvm.icmp "eq" %5, %2 : !llvm.ptr
    %7 = scf.if %6 -> (!llvm.ptr) {
      %8 = scf.if %arg2 -> (!llvm.ptr) {
        %9 = func.call @gen() : () -> !llvm.ptr
        llvm.store %9, %0 : !llvm.ptr, !llvm.ptr
        scf.yield %9 : !llvm.ptr
      } else {
        scf.yield %5 : !llvm.ptr
      }
      scf.yield %8 : !llvm.ptr
    } else {
      scf.yield %5 : !llvm.ptr
    }
  }
  %4 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
  return %4 : !llvm.ptr
}

}

// CHECK:     func.func @_Z3runiPPc(%arg0: i1) -> !llvm.ptr {
// CHECK-NEXT:       %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:       %0 = llvm.alloca %c1_i64 x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK-NEXT:       %1 = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:       scf.if %arg0 {
// CHECK-NEXT:         %3 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:         %4 = llvm.icmp "eq" %3, %1 : !llvm.ptr
// CHECK-NEXT:         %5 = scf.if %4 -> (!llvm.ptr) {
// CHECK-NEXT:           %6 = scf.if %arg0 -> (!llvm.ptr) {
// CHECK-NEXT:             %7 = func.call @gen() : () -> !llvm.ptr
// CHECK-NEXT:             llvm.store %7, %0 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:             scf.yield %7 : !llvm.ptr
// CHECK-NEXT:           } else {
// CHECK-NEXT:             scf.yield %3 : !llvm.ptr
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %6 : !llvm.ptr
// CHECK-NEXT:         } else {
// CHECK-NEXT:           scf.yield %3 : !llvm.ptr
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       %2 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:       return %2 : !llvm.ptr

