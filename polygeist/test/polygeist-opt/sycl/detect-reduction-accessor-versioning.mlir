// RUN: polygeist-opt --detect-reduction --split-input-file %s | FileCheck %s

// Original loop:
// for(size_t i = 0; i < 8; i++)
//   A[0] += B[i];
// Optimized loop:
// if (&A[A.get_range()] <= &B[0] || &A[0] >= &B[B.get_range()]) {
//   a = A[0];
//   for(size_t i = 0; i < 8; i++) 
//     a += B[i];
//   A[0] = a;
// } else 
//   for(size_t i = 0; i < 8; i++) 
//     A[0] += B[i];
// Note: The version condition implementation doesn't do short-circuit evaluation.

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_rw_dev = !sycl.accessor<[1, i32, read_write, device], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xi32, 1>)>)>
!sycl_accessor_1_i32_r_dev = !sycl.accessor<[1, i32, read, device], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xi32, 1>)>)>


// CHECK-LABEL: func.func private @test(
// CHECK-SAME:    %arg0: memref<?x!sycl_accessor_1_i32_rw_dev, 4>, %arg1: memref<?x!sycl_accessor_1_i32_r_dev, 4>) {

// COM: Obtain a pointer to the beginning of the accessor %arg0.
// CHECK:         %[[VAL_19:.*]] = sycl.id.constructor(%c0{{.*}}) : (index) -> memref<1x!sycl_id_1_>
// CHECK-NEXT:    [[ARG0_BEGIN:%.*]] = sycl.accessor.subscript %arg0[%2] : (memref<?x!sycl_accessor_1_i32_rw_dev, 4>, memref<1x!sycl_id_1_>) -> memref<?xi32, 1>

// COM: Obtain a pointer to the end of the accessor %arg0.
// CHECK-NEXT:    %[[VAL_21:.*]] = sycl.accessor.get_range(%arg0) : (memref<?x!sycl_accessor_1_i32_rw_dev, 4>) -> !sycl_range_1_
// CHECK-NEXT:    %[[VAL_22:.*]] = memref.alloca() : memref<1x!sycl_range_1_>
// CHECK-NEXT:    %[[VAL_23:.*]] = arith.constant 0 : index
// CHECK-NEXT:    memref.store %[[VAL_21]], %[[VAL_22]]{{\[}}%[[VAL_23]]] : memref<1x!sycl_range_1_>
// CHECK-NEXT:    %[[VAL_24:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[VAL_25:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[VAL_26:.*]] = sycl.range.get %[[VAL_22]]{{\[}}%[[VAL_25]]] : (memref<1x!sycl_range_1_>, i32) -> index
// CHECK-NEXT:    %[[VAL_27:.*]] = sycl.id.constructor(%[[VAL_26]]) : (index) -> memref<1x!sycl_id_1_>
// CHECK-NEXT:    [[ARG0_END:%.*]] = sycl.accessor.subscript %arg0{{\[}}%[[VAL_27]]] : (memref<?x!sycl_accessor_1_i32_rw_dev, 4>, memref<1x!sycl_id_1_>) -> memref<?xi32, 1>

// CHECK:         [[ARG1_BEGIN:%.*]] = sycl.accessor.subscript %arg1[%{{.*}}]
// CHECK:         [[ARG1_END:%.*]] = sycl.accessor.subscript %arg1[%{{.*}}]
// CHECK-DAG:     [[ARG0_END_PTR:%.*]] = "polygeist.memref2pointer"([[ARG0_END]]) : (memref<?xi32, 1>) -> !llvm.ptr<1>
// CHECK-DAG:     [[ARG1_BEGIN_PTR:%.*]] = "polygeist.memref2pointer"([[ARG1_BEGIN]]) : (memref<?xi32, 1>) -> !llvm.ptr<1>
// CHECK-NEXT:    [[BEFORE_COND:%.*]] = llvm.icmp "ule" [[ARG0_END_PTR]], [[ARG1_BEGIN_PTR]] : !llvm.ptr<1>
// CHECK-DAG:     [[ARG0_BEGIN_PTR:%.*]] = "polygeist.memref2pointer"([[ARG0_BEGIN]]) : (memref<?xi32, 1>) -> !llvm.ptr<1>
// CHECK-DAG:     [[ARG1_END_PTR:%.*]] = "polygeist.memref2pointer"([[ARG1_END]]) : (memref<?xi32, 1>) -> !llvm.ptr<1>
// CHECK-NEXT:    [[AFTER_COND:%.*]] = llvm.icmp "uge" [[ARG0_BEGIN_PTR]], [[ARG1_END_PTR]] : !llvm.ptr<1>

// COM: Version with condition: [[ARG0_END]] <= [[ARG1_BEGIN]] || [[ARG0_BEGIN]] >= [[ARG1_END]].
// CHECK-NEXT:    [[COND:%.*]] = arith.ori [[BEFORE_COND]], [[AFTER_COND]] : i1
// CHECK-NEXT:    scf.if [[COND]] {
// CHECK-NEXT:      %21 = affine.load %1[0] : memref<?xi32, 4>
// CHECK-NEXT:      %22 = scf.for %arg2 = %c0 to %c8 step %c1 iter_args(%arg3 = %21) -> (i32) {
// CHECK:             %27 = arith.addi %arg3, %26 : i32
// CHECK-NEXT:        scf.yield %27 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.store %22, %1[0] : memref<?xi32, 4>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.for %arg2 = %c0 to %c8 step %c1 {
// CHECK:             %25 = affine.load %1[0] : memref<?xi32, 4>
// CHECK-NEXT:        %26 = arith.addi %25, %24 : i32
// CHECK-NEXT:        affine.store %26, %1[0] : memref<?xi32, 4>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

func.func private @test(%arg0: memref<?x!sycl_accessor_1_i32_rw_dev, 4>, %arg1: memref<?x!sycl_accessor_1_i32_r_dev, 4>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0_i64 = arith.constant 0 : i64
  %alloca = memref.alloca() : memref<1x!sycl_id_1_>
  %cast = memref.cast %alloca : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
  %alloca_0 = memref.alloca() : memref<1x!sycl_id_1_>
  %cast_1 = memref.cast %alloca_0 : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
  %alloca_2 = memref.alloca() : memref<1x!sycl_id_1_>
  %cast_3 = memref.cast %alloca_2 : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
  %alloca_4 = memref.alloca() : memref<1x!sycl_id_1_>
  %cast_5 = memref.cast %alloca_4 : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
  %memspacecast = memref.memory_space_cast %cast_5 : memref<?x!sycl_id_1_> to memref<?x!sycl_id_1_, 4>
  %memspacecast_6 = memref.memory_space_cast %cast_1 : memref<?x!sycl_id_1_> to memref<?x!sycl_id_1_, 4>
  sycl.constructor @id(%memspacecast_6, %c0_i64) {MangledFunctionName = @_ZN4sycl3_V12idILi1EEC1ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE} : (memref<?x!sycl_id_1_, 4>, i64)
  %0 = affine.load %alloca_0[0] : memref<1x!sycl_id_1_>
  affine.store %0, %alloca[0] : memref<1x!sycl_id_1_>
  %1 = sycl.accessor.subscript %arg0[%cast] : (memref<?x!sycl_accessor_1_i32_rw_dev, 4>, memref<?x!sycl_id_1_>) -> memref<?xi32, 4>
  scf.for %i = %c0 to %c8 step %c1 {
    %2 = arith.index_cast %i : index to i64
    sycl.constructor @id(%memspacecast, %2) {MangledFunctionName = @_ZN4sycl3_V12idILi1EEC1ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE} : (memref<?x!sycl_id_1_, 4>, i64)
    %3 = affine.load %alloca_4[0] : memref<1x!sycl_id_1_>
    affine.store %3, %alloca_2[0] : memref<1x!sycl_id_1_>
    %4 = sycl.accessor.subscript %arg1[%cast_3] : (memref<?x!sycl_accessor_1_i32_r_dev, 4>, memref<?x!sycl_id_1_>) -> memref<?xi32, 4>
    %5 = affine.load %4[0] : memref<?xi32, 4>
    %6 = affine.load %1[0] : memref<?xi32, 4>
    %7 = arith.addi %6, %5 : i32
    affine.store %7, %1[0] : memref<?xi32, 4>
  }
  return
}
