// RUN: polygeist-opt --canonicalize --split-input-file %s | FileCheck %s

#set0 = affine_set<(d0) : (d0 == 0)>
#set1 = affine_set<(d0) : (d0 mod 2 == 0)>
#set2 = affine_set<(d0) : (d0 mod 4 == 0)>
#set3 = affine_set<(d0) : (d0 mod 8 == 0)>
#set4 = affine_set<(d0) : (d0 mod 16 == 0)>
module {
  func.func @bpnn_train_cuda(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: index, %arg4: index, %arg5: index) {
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %0 = arith.muli %arg3, %c16 : index
    affine.parallel (%arg6) = (0) to (symbol(%arg4)) {
      %1 = memref.alloca() : memref<16xf32>
      %2 = memref.alloca() : memref<16x16xf32>
      affine.parallel (%arg7, %arg8) = (0, 0) to (16, 16) {
        affine.if #set0(%arg7) {
          %10 = affine.load %arg1[%arg8 + %arg6 * 16 + 1] : memref<?xf32>
          affine.store %10, %1[%arg8] : memref<16xf32>
        }
        "polygeist.barrier"(%arg7, %arg8, %c0) : (index, index, index) -> ()
        %3 = affine.load %arg2[%arg7 + symbol(%arg3) + %arg6 * symbol(%0) + %arg8 * symbol(%arg3) + 1] : memref<?xf32>
        "polygeist.barrier"(%arg7, %arg8, %c0) : (index, index, index) -> ()
        %4 = affine.load %1[%arg8] : memref<16xf32>
        %5 = arith.mulf %3, %4 : f32
        affine.store %5, %2[%arg8, %arg7] : memref<16x16xf32>
        "polygeist.barrier"(%arg7, %arg8, %c0) : (index, index, index) -> ()
        %6 = affine.if #set1(%arg8) -> f32 {
          %10 = affine.load %2[%arg8 + 1, %arg7] : memref<16x16xf32>
          %11 = arith.addf %5, %10 : f32
          affine.store %11, %2[%arg8, %arg7] : memref<16x16xf32>
          affine.yield %11 : f32
        } else {
          affine.yield %5 : f32
        }
        "polygeist.barrier"(%arg7, %arg8, %c0) : (index, index, index) -> ()
        %7 = affine.if #set2(%arg8) -> f32 {
          %10 = affine.load %2[%arg8 + 2, %arg7] : memref<16x16xf32>
          %11 = arith.addf %6, %10 : f32
          affine.store %11, %2[%arg8, %arg7] : memref<16x16xf32>
          affine.yield %11 : f32
        } else {
          affine.yield %6 : f32
        }
        "polygeist.barrier"(%arg7, %arg8, %c0) : (index, index, index) -> ()
        %8 = affine.if #set3(%arg8) -> f32 {
          %10 = affine.load %2[%arg8 + 4, %arg7] : memref<16x16xf32>
          %11 = arith.addf %7, %10 : f32
          affine.store %11, %2[%arg8, %arg7] : memref<16x16xf32>
          affine.yield %11 : f32
        } else {
          affine.yield %7 : f32
        }
        "polygeist.barrier"(%arg7, %arg8, %c0) : (index, index, index) -> ()
        %9 = affine.if #set4(%arg8) -> f32 {
          %10 = affine.load %2[%arg8 + 8, %arg7] : memref<16x16xf32>
          %11 = arith.addf %8, %10 : f32
          affine.store %11, %2[%arg8, %arg7] : memref<16x16xf32>
          affine.yield %11 : f32
        } else {
          affine.yield %8 : f32
        }
        "polygeist.barrier"(%arg7, %arg8, %c0) : (index, index, index) -> ()
        affine.store %9, %arg2[%arg7 + symbol(%arg3) + %arg6 * symbol(%0) + %arg8 * symbol(%arg3) + 1] : memref<?xf32>
        "polygeist.barrier"(%arg7, %arg8, %c0) : (index, index, index) -> ()
        affine.if #set0(%arg7) {
          %10 = affine.load %2[%arg7, %arg8] : memref<16x16xf32>
          affine.store %10, %arg0[%arg8 + %arg6 * symbol(%arg5)] : memref<?xf32>
        }
      }
    }
    return
  }
}

// CHECK:   func.func @bpnn_train_cuda(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: index, %arg4: index, %arg5: index) {
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-DAG:     %c16 = arith.constant 16 : index
// CHECK-NEXT:     %0 = arith.muli %arg3, %c16 : index
// CHECK-NEXT:     affine.parallel (%arg6) = (0) to (symbol(%arg4)) {
// CHECK-NEXT:       %alloca = memref.alloca() : memref<16xf32>
// CHECK-NEXT:       %alloca_0 = memref.alloca() : memref<16x16xf32>
// CHECK-NEXT:       affine.parallel (%arg7, %arg8) = (0, 0) to (16, 16) {
// CHECK-NEXT:         affine.if #set0(%arg7) {
// CHECK-NEXT:           %8 = affine.load %arg1[%arg8 + %arg6 * 16 + 1] : memref<?xf32>
// CHECK-NEXT:           affine.store %8, %alloca[%arg8] : memref<16xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         %1 = affine.load %arg2[%arg7 + symbol(%arg3) + %arg6 * symbol(%0) + %arg8 * symbol(%arg3) + 1] : memref<?xf32>
// CHECK-NEXT:         "polygeist.barrier"(%arg7, %arg8, %c0) : (index, index, index) -> ()
// CHECK-NEXT:         %2 = affine.load %alloca[%arg8] : memref<16xf32>
// CHECK-NEXT:         %3 = arith.mulf %1, %2 : f32
// CHECK-NEXT:         affine.store %3, %alloca_0[%arg8, %arg7] : memref<16x16xf32>
// CHECK-NEXT:         "polygeist.barrier"(%arg7, %arg8, %c0) : (index, index, index) -> ()
// CHECK-NEXT:         %4 = affine.if #set1(%arg8) -> f32 {
// CHECK-NEXT:           %8 = affine.load %alloca_0[%arg8 + 1, %arg7] : memref<16x16xf32>
// CHECK-NEXT:           %9 = arith.addf %3, %8 : f32
// CHECK-NEXT:           affine.store %9, %alloca_0[%arg8, %arg7] : memref<16x16xf32>
// CHECK-NEXT:           affine.yield %9 : f32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           affine.yield %3 : f32
// CHECK-NEXT:         }
// CHECK-NEXT:         "polygeist.barrier"(%arg7, %arg8, %c0) : (index, index, index) -> ()
// CHECK-NEXT:         %5 = affine.if #set2(%arg8) -> f32 {
// CHECK-NEXT:           %8 = affine.load %alloca_0[%arg8 + 2, %arg7] : memref<16x16xf32>
// CHECK-NEXT:           %9 = arith.addf %4, %8 : f32
// CHECK-NEXT:           affine.store %9, %alloca_0[%arg8, %arg7] : memref<16x16xf32>
// CHECK-NEXT:           affine.yield %9 : f32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           affine.yield %4 : f32
// CHECK-NEXT:         }
// CHECK-NEXT:         "polygeist.barrier"(%arg7, %arg8, %c0) : (index, index, index) -> ()
// CHECK-NEXT:         %6 = affine.if #set3(%arg8) -> f32 {
// CHECK-NEXT:           %8 = affine.load %alloca_0[%arg8 + 4, %arg7] : memref<16x16xf32>
// CHECK-NEXT:           %9 = arith.addf %5, %8 : f32
// CHECK-NEXT:           affine.store %9, %alloca_0[%arg8, %arg7] : memref<16x16xf32>
// CHECK-NEXT:           affine.yield %9 : f32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           affine.yield %5 : f32
// CHECK-NEXT:         }
// CHECK-NEXT:         "polygeist.barrier"(%arg7, %arg8, %c0) : (index, index, index) -> ()
// CHECK-NEXT:         %7 = affine.if #set4(%arg8) -> f32 {
// CHECK-NEXT:           %8 = affine.load %alloca_0[%arg8 + 8, %arg7] : memref<16x16xf32>
// CHECK-NEXT:           %9 = arith.addf %6, %8 : f32
// CHECK-NEXT:           affine.store %9, %alloca_0[%arg8, %arg7] : memref<16x16xf32>
// CHECK-NEXT:           affine.yield %9 : f32
// CHECK-NEXT:         } else {
// CHECK-NEXT:           affine.yield %6 : f32
// CHECK-NEXT:         }
// CHECK-NEXT:         "polygeist.barrier"(%arg7, %arg8, %c0) : (index, index, index) -> ()
// CHECK-NEXT:         affine.store %7, %arg2[%arg7 + symbol(%arg3) + %arg6 * symbol(%0) + %arg8 * symbol(%arg3) + 1] : memref<?xf32>
// CHECK-NEXT:         "polygeist.barrier"(%arg7, %arg8, %c0) : (index, index, index) -> ()
// CHECK-NEXT:         affine.if #set0(%arg7) {
// CHECK-NEXT:           %8 = affine.load %alloca_0[%arg7, %arg8] : memref<16x16xf32>
// CHECK-NEXT:           affine.store %8, %arg0[%arg8 + %arg6 * symbol(%arg5)] : memref<?xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
