// RUN: polygeist-opt --polygeist-mem2reg --split-input-file %s | FileCheck %s

module {
  func.func @ll(%arg0: i16) -> i16 {
    %1 = memref.alloca() : memref<1x1xi16>
    affine.store %arg0, %1[0, 0] : memref<1x1xi16>
    %4 = affine.load %1[0, 0] : memref<1x1xi16>
    return %4 : i16
  }
}

// CHECK:   func.func @ll(%arg0: i16) -> i16 {
// CHECK-NEXT:     return %arg0 : i16
// CHECK-NEXT:   }
