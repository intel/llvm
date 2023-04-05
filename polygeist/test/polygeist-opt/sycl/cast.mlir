// RUN: polygeist-opt --convert-polygeist-to-llvm='use-opaque-pointers=1' --split-input-file %s | FileCheck %s

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>

// CHECK-LABEL:   llvm.func @test1(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.ptr {
// CHECK:           llvm.return %[[VAL_0]] : !llvm.ptr
// CHECK:         }

func.func @test1(%arg0: memref<?x!sycl_range_1_>) -> memref<?x!sycl_array_1_> {
  %0 = sycl.cast %arg0 : memref<?x!sycl_range_1_> to memref<?x!sycl_array_1_>
  func.return %0 : memref<?x!sycl_array_1_>
}

// -----

// CHECK-LABEL:   llvm.func @test2(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.ptr {
// CHECK:           llvm.return %[[VAL_0]] : !llvm.ptr
// CHECK:         }

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
func.func @test2(%arg0: memref<?x!sycl_id_1_>) -> memref<?x!sycl_array_1_> {
  %0 = sycl.cast %arg0 : memref<?x!sycl_id_1_> to memref<?x!sycl_array_1_>
  func.return %0: memref<?x!sycl_array_1_>
}

// -----

// CHECK-LABEL:   llvm.func @test_addrspaces(
// CHECK-SAME:                               %[[VAL_0:.*]]: !llvm.ptr<4>) -> !llvm.ptr<4> {
// CHECK:           llvm.return %[[VAL_0]] : !llvm.ptr<4>
// CHECK:         }

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
func.func @test_addrspaces(%arg0: memref<?x!sycl_id_1_, 4>) -> memref<?x!sycl_array_1_, 4> {
  %0 = sycl.cast %arg0 : memref<?x!sycl_id_1_, 4> to memref<?x!sycl_array_1_, 4>
  func.return %0: memref<?x!sycl_array_1_, 4>
}
