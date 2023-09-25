// RUN: polygeist-opt -licm -licm-enable-sycl-accessor-versioning --split-input-file %s | FileCheck %s --check-prefixes=CHECK,1PAIR
// RUN: polygeist-opt -licm -licm-enable-sycl-accessor-versioning -licm-sycl-accessor-pairs-limit=2 --split-input-file %s | FileCheck %s --check-prefixes=CHECK,2PAIRS

// Original loop:
// for(size_t i = 0; i < 8; i++)
//   A[i] += B[0];
// Optimized loop:
// if (0 < 8) {
//   if (&A[A.get_range()] <= &B[0] || &A[0] >= &B[B.get_range()]) {
//     b = B[0];
//     for(size_t i = 0; i < 8; i++) 
//       A[i] += b;
//   } else 
//     for(size_t i = 0; i < 8; i++) 
//       A[i] += b;
// }
// Note: The version condition implementation doesn't do short-circuit evaluation.

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_rw_dev = !sycl.accessor<[1, i32, read_write, device], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xi32, 1>)>)>
!sycl_accessor_1_i32_r_dev = !sycl.accessor<[1, i32, read, device], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xi32, 1>)>)>

// CHECK-LABEL: testSCFLoopVersioning
// CHECK-SAME:  ([[ARG0:%.*]]: memref<?x[[ACC_RW:!sycl_accessor_1_i32_rw_dev]], 4>, [[ARG1:%.*]]: memref<?x[[ACC_R:!sycl_accessor_1_i32_r_dev]], 4>) 

// CHECK-DAG:  [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:  [[C8:%.*]] = arith.constant 8 : index
// CHECK:      [[GUARD_COND:%.*]] = arith.cmpi slt, [[C0]], [[C8]] : index
// CHECK-NEXT: scf.if [[GUARD_COND]] {
// CHECK: [[ARG1_ACC:%.*]] = sycl.accessor.subscript %arg1[{{.*}}] : (memref<?x!sycl_accessor_1_i32_r_dev, 4>, memref<?x!sycl_id_1_>) -> memref<?xi32, 4>

// COM: Obtain a pointer to the beginning of the accessor %arg1.
// CHECK-DAG:  [[C0_index:%.*]] = arith.constant 0 : index
// CHECK-DAG:  [[ID_ALLOCA:%.*]] = sycl.id.constructor([[C0_index]]) : (index) -> memref<1x[[ID:!sycl_id_1_]]>
// CHECK-NEXT: [[ARG1_BEGIN:%.*]] = sycl.accessor.subscript [[ARG1]][[[ID_ALLOCA]]] : (memref<?x[[ACC_R]], 4>, memref<1x[[ID]]>) -> memref<?xi32, 1>

// COM: Obtain a pointer to the end of the accessor %arg1.
// CHECK-DAG:  [[RANGE_ALLOCA:%.*]] = memref.alloca() : memref<1x[[RANGE:!sycl_range_1_]]>
// CHECK-DAG:  [[C0_index:%.*]] = arith.constant 0 : index
// CHECK-DAG:  [[GET_RANGE:%.*]] = sycl.accessor.get_range([[ARG1]]) : (memref<?x[[ACC_R]], 4>) -> [[RANGE]]
// CHECK-NEXT: memref.store [[GET_RANGE]], [[RANGE_ALLOCA]][[[C0_index]]] : memref<1x[[RANGE]]>
// CHECK-DAG:  [[C0_i32:%.*]] = arith.constant 0 : i32
// CHECK-DAG:  [[C1_index:%.*]] = arith.constant 1 : index
// CHECK-NEXT: [[RANGE_GET:%.*]] = sycl.range.get [[RANGE_ALLOCA]][[[C0_i32]]] : (memref<1x[[RANGE]]>, i32) -> index
// CHECK-NEXT: [[ID_ALLOCA:%.*]] = sycl.id.constructor(%6) : (index) -> memref<1x[[ID:!sycl_id_1_]]>
// CHECK-NEXT: [[ARG1_END:%.*]] = sycl.accessor.subscript [[ARG1]][[[ID_ALLOCA]]] : (memref<?x[[ACC_R]], 4>, memref<1x[[ID]]>) -> memref<?xi32, 1>

// CHECK: [[ARG0_BEGIN:%.*]] = sycl.accessor.subscript [[ARG0]][{{.*}}] : (memref<?x[[ACC_RW]], 4>, memref<1x[[ID]]>) -> memref<?xi32, 1>
// CHECK: [[ARG0_END:%.*]] = sycl.accessor.subscript [[ARG0]][{{.*}}] : (memref<?x[[ACC_RW]], 4>, memref<1x[[ID]]>) -> memref<?xi32, 1>
// CHECK-DAG:  [[ARG1_END_PTR:%.*]] = "polygeist.memref2pointer"([[ARG1_END]]) : (memref<?xi32, 1>) -> !llvm.ptr<1>
// CHECK-DAG:  [[ARG0_BEGIN_PTR:%.*]] = "polygeist.memref2pointer"([[ARG0_BEGIN]]) : (memref<?xi32, 1>) -> !llvm.ptr<1>
// CHECK-NEXT: [[BEFORE_COND:%.*]] = llvm.icmp "ule" [[ARG1_END_PTR]], [[ARG0_BEGIN_PTR]] : !llvm.ptr<1>
// CHECK-DAG:  [[ARG1_BEGIN_PTR:%.*]] = "polygeist.memref2pointer"([[ARG1_BEGIN]]) : (memref<?xi32, 1>) -> !llvm.ptr<1>
// CHECK-DAG:  [[ARG0_END_PTR:%.*]] = "polygeist.memref2pointer"([[ARG0_END]]) : (memref<?xi32, 1>) -> !llvm.ptr<1>
// CHECK-NEXT: [[AFTER_COND:%.*]] = llvm.icmp "uge" [[ARG1_BEGIN_PTR]], [[ARG0_END_PTR]] : !llvm.ptr<1>

// COM: Version with condition: [[ARG1_END]] <= [[ARG0_BEGIN]] || [[ARG1_BEGIN]] >= [[ARG0_END]].
// CHECK-NEXT: [[COND:%.*]] = arith.ori [[BEFORE_COND]], [[AFTER_COND]] : i1
// CHECK-NEXT: scf.if [[COND]] {

// COM: Load of %arg1 accessor can be hoisted.
// CHECK: %{{.*}} = affine.load [[ARG1_ACC]][0] : memref<?xi32, 4>
// CHECK: scf.for
// CHECK: } else {
// CHECK-NEXT: scf.for

func.func private @testSCFLoopVersioning(%arg0: memref<?x!sycl_accessor_1_i32_rw_dev, 4>, %arg1: memref<?x!sycl_accessor_1_i32_r_dev, 4>) {
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
  scf.for %i = %c0 to %c8 step %c1 {
    %2 = arith.index_cast %i : index to i64
    sycl.constructor @id(%memspacecast, %c0_i64) {MangledFunctionName = @_ZN4sycl3_V12idILi1EEC1ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE} : (memref<?x!sycl_id_1_, 4>, i64)
    %3 = affine.load %alloca_4[0] : memref<1x!sycl_id_1_>
    affine.store %3, %alloca_2[0] : memref<1x!sycl_id_1_>
    %4 = sycl.accessor.subscript %arg1[%cast_3] : (memref<?x!sycl_accessor_1_i32_r_dev, 4>, memref<?x!sycl_id_1_>) -> memref<?xi32, 4>
    %5 = affine.load %4[0] : memref<?xi32, 4>
    sycl.constructor @id(%memspacecast_6, %2) {MangledFunctionName = @_ZN4sycl3_V12idILi1EEC1ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE} : (memref<?x!sycl_id_1_, 4>, i64)
    %6 = affine.load %alloca_0[0] : memref<1x!sycl_id_1_>
    affine.store %6, %alloca[0] : memref<1x!sycl_id_1_>
    %7 = sycl.accessor.subscript %arg0[%cast] : (memref<?x!sycl_accessor_1_i32_rw_dev, 4>, memref<?x!sycl_id_1_>) -> memref<?xi32, 4>
    %8 = affine.load %7[0] : memref<?xi32, 4>
    %9 = arith.addi %8, %5 : i32
    affine.store %9, %7[0] : memref<?xi32, 4>
  }
  return
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_rw_dev = !sycl.accessor<[1, i32, read_write, device], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xi32, 1>)>)>
!sycl_accessor_1_i32_r_dev = !sycl.accessor<[1, i32, read, device], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xi32, 1>)>)>

// CHECK: [[GUARD_COND:#.*]] = affine_set<() : (7 >= 0)>

// CHECK-LABEL: testAffineLoopVersioning
// CHECK-SAME:  ([[ARG0:%.*]]: memref<?x[[ACC_RW:!sycl_accessor_1_i32_rw_dev]], 4>, [[ARG1:%.*]]: memref<?x[[ACC_R:!sycl_accessor_1_i32_r_dev]], 4>)

// CHECK: affine.if [[GUARD_COND]]() {
// CHECK: [[ARG1_ACC:%.*]] = sycl.accessor.subscript %arg1[{{.*}}] : (memref<?x!sycl_accessor_1_i32_r_dev, 4>, memref<?x!sycl_id_1_>) -> memref<?xi32, 4>

// COM: Version with condition: [[ARG1_END]] <= [[ARG0_BEGIN]] || [[ARG1_BEGIN]] >= [[ARG0_END]].
// CHECK:      [[BEFORE_COND:%.*]] = llvm.icmp "ule" {{.*}}, {{.*}} : !llvm.ptr<1>
// CHECK:      [[AFTER_COND:%.*]] = llvm.icmp "uge" {{.*}}, {{.*}} : !llvm.ptr<1>
// CHECK-NEXT: [[COND:%.*]] = arith.ori [[BEFORE_COND]], [[AFTER_COND]] : i1
// CHECK-NEXT: scf.if [[COND]] {

// COM: Load of %arg1 accessor can be hoisted.
// CHECK: %{{.*}} = affine.load [[ARG1_ACC]][0] : memref<?xi32, 4>
// CHECK: affine.for
// CHECK: } else {
// CHECK-NEXT: affine.for

func.func private @testAffineLoopVersioning(%arg0: memref<?x!sycl_accessor_1_i32_rw_dev, 4>, %arg1: memref<?x!sycl_accessor_1_i32_r_dev, 4>) {
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
  affine.for %i = 0 to 8 {
    %2 = arith.index_cast %i : index to i64
    sycl.constructor @id(%memspacecast, %c0_i64) {MangledFunctionName = @_ZN4sycl3_V12idILi1EEC1ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE} : (memref<?x!sycl_id_1_, 4>, i64)
    %3 = affine.load %alloca_4[0] : memref<1x!sycl_id_1_>
    affine.store %3, %alloca_2[0] : memref<1x!sycl_id_1_>
    %4 = sycl.accessor.subscript %arg1[%cast_3] : (memref<?x!sycl_accessor_1_i32_r_dev, 4>, memref<?x!sycl_id_1_>) -> memref<?xi32, 4>
    %5 = affine.load %4[0] : memref<?xi32, 4>
    sycl.constructor @id(%memspacecast_6, %2) {MangledFunctionName = @_ZN4sycl3_V12idILi1EEC1ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE} : (memref<?x!sycl_id_1_, 4>, i64)
    %6 = affine.load %alloca_0[0] : memref<1x!sycl_id_1_>
    affine.store %6, %alloca[0] : memref<1x!sycl_id_1_>
    %7 = sycl.accessor.subscript %arg0[%cast] : (memref<?x!sycl_accessor_1_i32_rw_dev, 4>, memref<?x!sycl_id_1_>) -> memref<?xi32, 4>
    %8 = affine.load %7[0] : memref<?xi32, 4>
    %9 = arith.addi %8, %5 : i32
    affine.store %9, %7[0] : memref<?xi32, 4>
  }
  return
}

// -----

// This test requires -licm-sycl-accessor-pairs-limit >= 2.
// Original loop:
// for(size_t i = 0; i < 8; i++) {
//   A[0] = 1;
//   B[i] = 2;
//   C[i] = 3;
// }
// Optimized loop:
// if (0 < 8) {
//   if ((&A[A.get_range()] <= &B[0] || &A[0] >= &B[B.get_range()])
//       && (&A[A.get_range()] <= &C[0] || &A[0] >= &C[C.get_range()])) {
//     A[0] = 1;
//     for(size_t i = 0; i < 8; i++) {
//       B[i] = 2;
//       C[i] = 3;
//     }
//   } else 
//     for(size_t i = 0; i < 8; i++) {
//       A[0] = 1;
//       B[i] = 2;
//       C[i] = 3;
//     }
// }
// Note: The version condition implementation doesn't do short-circuit evaluation.

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_w_dev = !sycl.accessor<[1, i32, write, device], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xi32, 1>)>)>

// COM: There is only one loop, i.e., the loop is not versioned.
// 1PAIR-LABEL: testSCFLoopVersioning
// 1PAIR-SAME:  ([[ARG0:%.*]]: memref<?x[[ACC_W:!sycl_accessor_1_i32_w_dev]], 4>, [[ARG1:%.*]]: memref<?x[[ACC_W]], 4>, [[ARG1:%.*]]: memref<?x[[ACC_W]], 4>) 
// 1PAIR: [[C1_i32:%.*]] = arith.constant 1 : i32
// 1PAIR: [[ARG0_ACC:%.*]] = sycl.accessor.subscript %arg0[{{.*}}] : (memref<?x!sycl_accessor_1_i32_w_dev, 4>, memref<?x!sycl_id_1_>) -> memref<?xi32, 4>
// 1PAIR: scf.for
// 1PAIR: affine.store [[C1_i32]], [[ARG0_ACC]][0] : memref<?xi32, 4>
// 1PAIR-NOT: } else {
// 1PAIR-NOT: scf.for

// 2PAIRS-LABEL: testSCFLoopVersioning
// 2PAIRS-SAME:  ([[ARG0:%.*]]: memref<?x[[ACC_W:!sycl_accessor_1_i32_w_dev]], 4>, [[ARG1:%.*]]: memref<?x[[ACC_W]], 4>, [[ARG1:%.*]]: memref<?x[[ACC_W]], 4>) 

// 2PAIRS: [[C1_i32:%.*]] = arith.constant 1 : i32
// 2PAIRS: [[ARG0_ACC:%.*]] = sycl.accessor.subscript %arg0[{{.*}}] : (memref<?x!sycl_accessor_1_i32_w_dev, 4>, memref<?x!sycl_id_1_>) -> memref<?xi32, 4>

// COM: Version with condition: ([[ARG0_END]] <= [[ARG1_BEGIN]] || [[ARG0_BEGIN]] >= [[ARG1_END]])
// COM:                          && ([[ARG0_END]] <= [[ARG2_BEGIN]] || [[ARG0_BEGIN]] >= [[ARG2_END]]).
// 2PAIRS:      [[BEFORE_COND1:%.*]] = llvm.icmp "ule" {{.*}}, {{.*}} : !llvm.ptr<1>
// 2PAIRS:      [[AFTER_COND1:%.*]] = llvm.icmp "uge" {{.*}}, {{.*}} : !llvm.ptr<1>
// 2PAIRS-NEXT: [[COND1:%.*]] = arith.ori [[BEFORE_COND1]], [[AFTER_COND1]] : i1
// 2PAIRS:      [[BEFORE_COND2:%.*]] = llvm.icmp "ule" {{.*}}, {{.*}} : !llvm.ptr<1>
// 2PAIRS:      [[AFTER_COND2:%.*]] = llvm.icmp "uge" {{.*}}, {{.*}} : !llvm.ptr<1>
// 2PAIRS-NEXT: [[COND2:%.*]] = arith.ori [[BEFORE_COND2]], [[AFTER_COND2]] : i1
// 2PAIRS-NEXT: [[COND:%.*]] = arith.andi [[COND1]], [[COND2]] : i1
// 2PAIRS-NEXT: scf.if [[COND]] {

// COM: Store to %arg0 accessor can be hoisted.
// 2PAIRS: affine.store [[C1_i32]], [[ARG0_ACC]][0] : memref<?xi32, 4>
// 2PAIRS: scf.for
// 2PAIRS: } else {
// 2PAIRS-NEXT: scf.for

func.func private @testSCFLoopVersioning(%arg0: memref<?x!sycl_accessor_1_i32_w_dev, 4>, %arg1: memref<?x!sycl_accessor_1_i32_w_dev, 4>, %arg2: memref<?x!sycl_accessor_1_i32_w_dev, 4>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c3_i32 = arith.constant 3 : i32
  %c0_i64 = arith.constant 0 : i64
  %alloca = memref.alloca() : memref<1x!sycl_id_1_>
  %cast = memref.cast %alloca : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
  %alloca_0 = memref.alloca() : memref<1x!sycl_id_1_>
  %cast_1 = memref.cast %alloca_0 : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
  %alloca_2 = memref.alloca() : memref<1x!sycl_id_1_>
  %cast_3 = memref.cast %alloca_2 : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
  %alloca_4 = memref.alloca() : memref<1x!sycl_id_1_>
  %cast_5 = memref.cast %alloca_4 : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
  %alloca_6 = memref.alloca() : memref<1x!sycl_id_1_>
  %cast_7 = memref.cast %alloca_6 : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
  %alloca_8 = memref.alloca() : memref<1x!sycl_id_1_>
  %cast_9 = memref.cast %alloca_8 : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
  %memspacecast = memref.memory_space_cast %cast_9 : memref<?x!sycl_id_1_> to memref<?x!sycl_id_1_, 4>
  %memspacecast_10 = memref.memory_space_cast %cast_5 : memref<?x!sycl_id_1_> to memref<?x!sycl_id_1_, 4>
  %memspacecast_11 = memref.memory_space_cast %cast_1 : memref<?x!sycl_id_1_> to memref<?x!sycl_id_1_, 4>
  scf.for %i = %c0 to %c8 step %c1 {
    %2 = arith.index_cast %i : index to i64
    sycl.constructor @id(%memspacecast, %c0_i64) {MangledFunctionName = @_ZN4sycl3_V12idILi1EEC1ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE} : (memref<?x!sycl_id_1_, 4>, i64)
    %3 = affine.load %alloca_8[0] : memref<1x!sycl_id_1_>
    affine.store %3, %alloca_6[0] : memref<1x!sycl_id_1_>
    %4 = sycl.accessor.subscript %arg0[%cast_7] : (memref<?x!sycl_accessor_1_i32_w_dev, 4>, memref<?x!sycl_id_1_>) -> memref<?xi32, 4>
    affine.store %c1_i32, %4[0] : memref<?xi32, 4>
    sycl.constructor @id(%memspacecast_10, %2) {MangledFunctionName = @_ZN4sycl3_V12idILi1EEC1ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE} : (memref<?x!sycl_id_1_, 4>, i64)
    %5 = affine.load %alloca_4[0] : memref<1x!sycl_id_1_>
    affine.store %5, %alloca_2[0] : memref<1x!sycl_id_1_>
    %6 = sycl.accessor.subscript %arg1[%cast_3] : (memref<?x!sycl_accessor_1_i32_w_dev, 4>, memref<?x!sycl_id_1_>) -> memref<?xi32, 4>
    affine.store %c2_i32, %6[0] : memref<?xi32, 4>
    sycl.constructor @id(%memspacecast_11, %2) {MangledFunctionName = @_ZN4sycl3_V12idILi1EEC1ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE} : (memref<?x!sycl_id_1_, 4>, i64)
    %7 = affine.load %alloca_0[0] : memref<1x!sycl_id_1_>
    affine.store %7, %alloca[0] : memref<1x!sycl_id_1_>
    %8 = sycl.accessor.subscript %arg2[%cast] : (memref<?x!sycl_accessor_1_i32_w_dev, 4>, memref<?x!sycl_id_1_>) -> memref<?xi32, 4>
    affine.store %c3_i32, %8[0] : memref<?xi32, 4>
  }
  return
}
