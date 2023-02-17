// RUN: polygeist-opt --cpuify="method=distribute" --split-input-file %s | FileCheck %s

module {
  func.func private @capture(%a : memref<i32>) 
  func.func private @use(%a : memref<?xi32>, %b : f32, %d : i32, %e : f32)
  func.func @main() {
    %c0 = arith.constant 0 : index
    %cc1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    scf.parallel (%arg2) = (%c0) to (%c5) step (%cc1) {
      %a1 = memref.alloca() : memref<2xi32>
      %a2 = memref.cast %a1 : memref<2xi32> to memref<?xi32>
      %b1 = memref.alloca() : memref<f32>
      %c1 = memref.alloca() : memref<i32>
      %d1 = memref.alloca() : memref<1xi32>
      %b2 = memref.load %b1[] : memref<f32>
      func.call @capture(%c1) : (memref<i32>) -> ()
      %d2 = memref.cast %d1 : memref<1xi32> to memref<?xi32>
      
      %e1 = memref.alloca() : memref<1xf32>
      %e2 = memref.cast %e1 : memref<1xf32> to memref<?xf32>
      %e3 = memref.load %e2[%c0] : memref<?xf32>

      "polygeist.barrier"(%arg2) : (index) -> ()
      
      %d3 = memref.load %d2[%c0] : memref<?xi32>
      func.call @use(%a2, %b2, %d3, %e3) : (memref<?xi32>, f32, i32, f32) -> ()
      scf.yield
    }
    return
  }
}

// CHECK:   func.func @main() {
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c5 = arith.constant 5 : index
// CHECK-NEXT:     memref.alloca_scope {
// CHECK-DAG:      %alloca = memref.alloca(%c5) : memref<?xf32>
// CHECK-DAG:      %alloca_0 = memref.alloca(%c5) : memref<?xf32>
// CHECK-DAG:      %alloca_1 = memref.alloca(%c5) : memref<?xmemref<?xi32>>
// CHECK-DAG:      %alloca_2 = memref.alloca(%c5) : memref<?x2xi32>
// CHECK-DAG:      %alloca_3 = memref.alloca(%c5) : memref<?xi32>
// CHECK-DAG:      %alloca_4 = memref.alloca(%c5) : memref<?x1xi32>
// CHECK-NEXT:     scf.parallel (%arg0) = (%c0) to (%c5) step (%c1) {
// CHECK-NEXT:       %0 = "polygeist.subindex"(%alloca_2, %arg0) : (memref<?x2xi32>, index) -> memref<2xi32>
// CHECK-NEXT:       %cast = memref.cast %0 : memref<2xi32> to memref<?xi32>
// CHECK-NEXT:       memref.store %cast, %alloca_1[%arg0] : memref<?xmemref<?xi32>>
// CHECK-NEXT:       %alloca_5 = memref.alloca() : memref<f32>
// CHECK-NEXT:       %1 = memref.load %alloca_5[] : memref<f32>
// CHECK-NEXT:       memref.store %1, %alloca_0[%arg0] : memref<?xf32>
// CHECK-NEXT:       %2 = "polygeist.subindex"(%alloca_3, %arg0) : (memref<?xi32>, index) -> memref<i32>
// CHECK-NEXT:       func.call @capture(%2) : (memref<i32>) -> ()
// CHECK-NEXT:       %alloca_6 = memref.alloca() : memref<1xf32>
// CHECK-NEXT:       %3 = memref.load %alloca_6[%c0] : memref<1xf32>
// CHECK-NEXT:       memref.store %3, %alloca[%arg0] : memref<?xf32>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.parallel (%arg0) = (%c0) to (%c5) step (%c1) {
// CHECK-DAG:        %[[i9:.+]] = memref.load %alloca_1[%arg0] : memref<?xmemref<?xi32>>
// CHECK-DAG:        %[[i10:.+]] = memref.load %alloca_0[%arg0] : memref<?xf32>
// CHECK-DAG:        %[[i11:.+]] = memref.load %alloca[%arg0] : memref<?xf32>
// CHECK-NEXT:       %3 = "polygeist.subindex"(%alloca_4, %arg0) : (memref<?x1xi32>, index) -> memref<1xi32>
// CHECK-NEXT:       %[[i8:.+]] = memref.load %3[%c0] : memref<1xi32>
// CHECK-NEXT:       func.call @use(%[[i9]], %[[i10]], %[[i8]], %[[i11]]) : (memref<?xi32>, f32, i32, f32) -> ()
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
