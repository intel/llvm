// RUN: polygeist-opt -licm -enable-sycl-accessor-versioning %s | FileCheck %s

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xi32, 1>)>)>
!sycl_accessor_1_i32_r_gb = !sycl.accessor<[1, i32, read, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xi32, 1>)>)>

// CHECK-LABEL: test
// CHECK-SAME:  ([[ARG0:%.*]]: memref<?x[[ACC_RW:!sycl_accessor_1_i32_rw_gb]], 4>, [[ARG1:%.*]]: memref<?x[[ACC_R:!sycl_accessor_1_i32_r_gb]], 4>) 

// CHECK-DAG:  [[C0:%.*]] = arith.constant 0 : index
// CHECK-DAG:  [[C8:%.*]] = arith.constant 8 : index
// CHECK:      [[GUARD_COND:%.*]] = arith.cmpi slt, [[C0]], [[C8]] : index
// CHECK-NEXT: scf.if [[GUARD_COND]] {

// COM: Obtain a pointer to the beginning of the accessor %arg1.
// CHECK-DAG:  [[ID_ALLOCA:%.*]] = memref.alloca() : memref<1x[[ID:!sycl_id_1_]]>
// CHECK-DAG:  [[C0_i32:%.*]] = arith.constant 0 : i32
// CHECK-DAG:  [[C0_i64:%.*]] = arith.constant 0 : i64
// CHECK-DAG:  [[C0_index:%.*]] = arith.constant 0 : index
// CHECK-NEXT: [[ID_GET:%.*]] = sycl.id.get [[ID_ALLOCA]][[[C0_i32]]] {ArgumentTypes = [memref<1x[[ID]]>, i32], FunctionName = @"operator[]", TypeName = @id} : (memref<1x[[ID]]>, i32) -> memref<?xi64>
// CHECK-NEXT: memref.store [[C0_i64]], [[ID_GET]][[[C0_index]]] : memref<?xi64>
// CHECK-NEXT: [[ARG1_BEGIN:%.*]] = sycl.accessor.subscript [[ARG1]][[[ID_ALLOCA]]] {ArgumentTypes = [memref<?x[[ACC_R]], 4>, memref<1x[[ID]]>], FunctionName = @"operator[]", TypeName = @accessor} : (memref<?x[[ACC_R]], 4>, memref<1x[[ID]]>) -> memref<?xi32, 1>

// COM: Obtain a pointer to the end of the accessor %arg1.
// CHECK-DAG:  [[RANGE_ALLOCA:%.*]] = memref.alloca() : memref<1x[[RANGE:!sycl_range_1_]]>
// CHECK-DAG:  [[C0_index:%.*]] = arith.constant 0 : index
// CHECK-DAG:  [[GET_RANGE:%.*]] = sycl.accessor.get_range([[ARG1]]) {ArgumentTypes = [memref<?x[[ACC_R]], 4>], FunctionName = @get_range, TypeName = @accessor} : (memref<?x[[ACC_R]], 4>) -> [[RANGE]]
// CHECK-NEXT: memref.store [[GET_RANGE]], [[RANGE_ALLOCA]][[[C0_index]]] : memref<1x[[RANGE]]>
// CHECK-DAG:  [[ID_ALLOCA:%.*]] = memref.alloca() : memref<1x[[ID:!sycl_id_1_]]>
// CHECK-DAG:  [[C0_i32:%.*]] = arith.constant 0 : i32
// CHECK-DAG:  [[C1_i64:%.*]] = arith.constant 1 : i64
// CHECK-NEXT: [[ID_GET:%.*]] = sycl.id.get [[ID_ALLOCA]][[[C0_i32]]] {ArgumentTypes = [memref<1x[[ID]]>, i32], FunctionName = @"operator[]", TypeName = @id} : (memref<1x[[ID]]>, i32) -> memref<?xi64>
// CHECK-NEXT: [[C0_i32:%.*]] = arith.constant 0 : i32
// CHECK-NEXT: [[RANGE_GET:%.*]] = sycl.range.get [[RANGE_ALLOCA]][[[C0_i32]]] {ArgumentTypes = [memref<1x[[RANGE]]>, i32], FunctionName = @get, TypeName = @range} : (memref<1x[[RANGE]]>, i32) -> i64
// CHECK-NEXT: [[SUB:%.*]] = arith.subi [[RANGE_GET]], [[C1_i64]] : i64
// CHECK-NEXT: memref.store [[SUB]], [[ID_GET]][[[C0_index]]] : memref<?xi64>
// CHECK-NEXT: [[ACC_SUB:%.*]] = sycl.accessor.subscript [[ARG1]][[[ID_ALLOCA]]] {ArgumentTypes = [memref<?x[[ACC_R]], 4>, memref<1x[[ID]]>], FunctionName = @"operator[]", TypeName = @accessor} : (memref<?x[[ACC_R]], 4>, memref<1x[[ID]]>) -> memref<?xi32, 1>
// CHECK-NEXT: [[C1_INDEX:%.*]] = arith.constant 1 : index
// CHECK-NEXT: [[ARG1_END:%.*]] = "polygeist.subindex"([[ACC_SUB]], [[C1_INDEX]]) : (memref<?xi32, 1>, index) -> memref<?xi32, 1>

// CHECK: [[ARG0_BEGIN:%.*]] = sycl.accessor.subscript [[ARG0]][{{.*}}] {ArgumentTypes = [memref<?x[[ACC_RW]], 4>, memref<1x[[ID]]>], FunctionName = @"operator[]", TypeName = @accessor} : (memref<?x[[ACC_RW]], 4>, memref<1x[[ID]]>) -> memref<?xi32, 1>
// CHECK: [[ARG0_END:%.*]] = "polygeist.subindex"({{.*}}, {{.*}}) : (memref<?xi32, 1>, index) -> memref<?xi32, 1>
// CHECK-DAG:  [[ARG1_END_PTR:%.*]] = "polygeist.memref2pointer"([[ARG1_END]]) : (memref<?xi32, 1>) -> !llvm.ptr<i32, 1>
// CHECK-DAG:  [[ARG0_BEGIN_PTR:%.*]] = "polygeist.memref2pointer"([[ARG0_BEGIN]]) : (memref<?xi32, 1>) -> !llvm.ptr<i32, 1>
// CHECK-NEXT: [[BEFORE_COND:%.*]] = llvm.icmp "ule" [[ARG1_END_PTR]], [[ARG0_BEGIN_PTR]] : !llvm.ptr<i32, 1>
// CHECK-DAG:  [[ARG1_BEGIN_PTR:%.*]] = "polygeist.memref2pointer"([[ARG1_BEGIN]]) : (memref<?xi32, 1>) -> !llvm.ptr<i32, 1>
// CHECK-DAG:  [[ARG0_END_PTR:%.*]] = "polygeist.memref2pointer"([[ARG0_END]]) : (memref<?xi32, 1>) -> !llvm.ptr<i32, 1>
// CHECK-NEXT: [[AFTER_COND:%.*]] = llvm.icmp "uge" [[ARG1_BEGIN_PTR]], [[ARG0_END_PTR]] : !llvm.ptr<i32, 1>

// COM: Version with condition: [[ARG1_END]] <= [[ARG0_BEGIN]] || [[ARG1_BEGIN]] >= [[ARG0_END]].
// CHECK-NEXT: [[COND:%.*]] = arith.ori [[BEFORE_COND]], [[AFTER_COND]] : i1
// CHECK-NEXT: scf.if [[COND]] {

// COM: Load of %arg1 accessor can be hoisted.
// CHECK: [[ARG1_ACC:%.*]] = sycl.accessor.subscript %arg1[{{.*}}] {ArgumentTypes = [memref<?x!sycl_accessor_1_i32_r_gb, 4>, memref<?x!sycl_id_1_>], FunctionName = @"operator[]", TypeName = @accessor} : (memref<?x!sycl_accessor_1_i32_r_gb, 4>, memref<?x!sycl_id_1_>) -> memref<?xi32, 4>
// CHECK: %{{.*}} = affine.load [[ARG1_ACC]][0] : memref<?xi32, 4>
// CHECK: scf.for
// CHECK: } else {
// CHECK-NEXT: scf.for

func.func private @test(%arg0: memref<?x!sycl_accessor_1_i32_rw_gb, 4>, %arg1: memref<?x!sycl_accessor_1_i32_r_gb, 4>) {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
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
    %2 = arith.index_cast %i : index to i32
    sycl.constructor @id(%memspacecast, %c0_i64) {MangledFunctionName = @_ZN4sycl3_V12idILi1EEC1ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE} : (memref<?x!sycl_id_1_, 4>, i64)
    %3 = affine.load %alloca_4[0] : memref<1x!sycl_id_1_>
    affine.store %3, %alloca_2[0] : memref<1x!sycl_id_1_>
    %4 = sycl.accessor.subscript %arg1[%cast_3] {ArgumentTypes = [memref<?x!sycl_accessor_1_i32_r_gb, 4>, memref<?x!sycl_id_1_>], FunctionName = @"operator[]", TypeName = @accessor} : (memref<?x!sycl_accessor_1_i32_r_gb, 4>, memref<?x!sycl_id_1_>) -> memref<?xi32, 4>
    %5 = affine.load %4[0] : memref<?xi32, 4>
    %6 = arith.extui %2 : i32 to i64
    sycl.constructor @id(%memspacecast_6, %6) {MangledFunctionName = @_ZN4sycl3_V12idILi1EEC1ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE} : (memref<?x!sycl_id_1_, 4>, i64)
    %7 = affine.load %alloca_0[0] : memref<1x!sycl_id_1_>
    affine.store %7, %alloca[0] : memref<1x!sycl_id_1_>
    %8 = sycl.accessor.subscript %arg0[%cast] {ArgumentTypes = [memref<?x!sycl_accessor_1_i32_rw_gb, 4>, memref<?x!sycl_id_1_>], FunctionName = @"operator[]", TypeName = @accessor} : (memref<?x!sycl_accessor_1_i32_rw_gb, 4>, memref<?x!sycl_id_1_>) -> memref<?xi32, 4>
    %9 = affine.load %8[0] : memref<?xi32, 4>
    %10 = arith.addi %9, %5 : i32
    affine.store %10, %8[0] : memref<?xi32, 4>
  }
  return
}
