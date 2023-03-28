// RUN: sycl-mlir-opt %s | sycl-mlir-opt | FileCheck %s
// RUN: sycl-mlir-opt %s --mlir-print-op-generic | sycl-mlir-opt | FileCheck %s

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
!sycl_array_2_ = !sycl.array<[2], (memref<2xi64, 4>)>
!sycl_array_3_ = !sycl.array<[3], (memref<3xi64, 4>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_id_2_ = !sycl.id<[2], (!sycl_array_2_)>
!sycl_id_3_ = !sycl.id<[3], (!sycl_array_3_)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
!sycl_range_2_ = !sycl.range<[2], (!sycl_array_2_)>
!sycl_range_3_ = !sycl.range<[3], (!sycl_array_3_)>
!sycl_accessor_1_i32_w_gb = !sycl.accessor<[1, i32, write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_ato_gb = !sycl.accessor<[1, i32, atomic, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 3>)>)>
!sycl_atomic_i32_1_ = !sycl.atomic<[i32, 1], (memref<?xi32, 1>)>

// CHECK-LABEL: test_addrspacecast_to_generic
func.func @test_addrspacecast_to_generic(%arg0: memref<?xi32>) -> memref<?xi32, 4> {
  %0 = sycl.addrspacecast %arg0 : memref<?xi32> to memref<?xi32, 4>
  return %0 : memref<?xi32, 4>
}

// CHECK-LABEL: test_addrspacecast_from_generic
func.func @test_addrspacecast_from_generic(%arg0: memref<?xi32, 4>) -> memref<?xi32> {
  %0 = sycl.addrspacecast %arg0 : memref<?xi32, 4> to memref<?xi32>
  return %0 : memref<?xi32>
}

// CHECK-LABEL: test_cast_id
func.func @test_cast_id(%arg: memref<1x!sycl_id_1_>) -> memref<1x!sycl_array_1_> {
  %0 = sycl.cast %arg : memref<1x!sycl_id_1_> to memref<1x!sycl_array_1_>
  return %0 : memref<1x!sycl_array_1_>
}

// CHECK-LABEL: test_cast_range
func.func @test_cast_range(%arg: memref<1x!sycl_range_2_>) -> memref<1x!sycl_array_2_> {
  %0 = sycl.cast %arg : memref<1x!sycl_range_2_> to memref<1x!sycl_array_2_>
  return %0 : memref<1x!sycl_array_2_>
}

// CHECK-LABEL: test_cast_accessor
func.func @test_cast_accessor(%arg: memref<1x!sycl_accessor_1_i32_w_gb>) -> memref<1x!sycl.accessor_common> {
  %0 = sycl.cast %arg : memref<1x!sycl_accessor_1_i32_w_gb> to memref<1x!sycl.accessor_common>
  return %0 : memref<1x!sycl.accessor_common>
}

// CHECK-LABEL: test_void_call
func.func @test_void_call() {
  sycl.call @foo() {MangledFunctionName = @foov, TypeName = @A} : () -> ()
  return
}

// CHECK-LABEL: test_void_call_with_arg
func.func @test_void_call_with_arg(%arg_0: i32) {
  sycl.call @foo(%arg_0) {MangledFunctionName = @fooi, TypeName = @A} : (i32) -> ()
  return
}

// CHECK-LABEL: test_i32_call
func.func @test_i32_call() {
  %0 = sycl.call @bar() {MangledFunctionName = @barv, TypeName = @A} : () -> (i32)
  return
}

// CHECK-LABEL: test_i32_call_with_arg
func.func @test_i32_call_with_arg(%arg_0: i32) {
  %0 = sycl.call @bar(%arg_0) {MangledFunctionName = @bari, TypeName = @A} : (i32) -> (i32)
  return
}

// CHECK-LABEL: test_i32_call_with_two_args
func.func @test_i32_call_with_two_args(%arg_0: i32, %arg_1: i64) {
  %0 = sycl.call @bar(%arg_0, %arg_1) {MangledFunctionName = @baril, TypeName = @A} : (i32, i64) -> (i32)
  return
}

// CHECK-LABEL: test_constructor
func.func @test_constructor(%id: memref<1x!sycl_id_1_>) {
  sycl.constructor @id(%id) {MangledFunctionName = @idv} : (memref<1x!sycl_id_1_>)
  return
}

// CHECK-LABEL: test_constructor_with_arg
func.func @test_constructor_with_arg(%range: memref<1x!sycl_range_1_>,
                                     %arg_0: i32) {
  sycl.constructor @range(%range, %arg_0) {MangledFunctionName = @rangei} : (memref<1x!sycl_range_1_>, i32)
  return
}

// CHECK-LABEL: test_constructor_with_args
func.func @test_constructor_with_args(%range: memref<1x!sycl_range_1_>,
                                      %arg_0: i32,
				      %arg_1: i32,
				      %arg_2: i32) {
  sycl.constructor @range(%range, %arg_0, %arg_1, %arg_2) {MangledFunctionName = @rangeiii} : (memref<1x!sycl_range_1_>, i32, i32, i32)
  return
}

// CHECK-LABEL: test_num_work_items
func.func @test_num_work_items() -> !sycl_range_1_ {
  %0 = sycl.num_work_items : !sycl_range_1_
  return %0 : !sycl_range_1_
}

// CHECK-LABEL: test_num_work_items_dim
func.func @test_num_work_items_dim(%i: i32) -> index {
  %0 = sycl.num_work_items %i : index
  return %0 : index
}

// CHECK-LABEL: test_global_id
func.func @test_global_id() -> !sycl_id_2_ {
  %0 = sycl.global_id : !sycl_id_2_
  return %0 : !sycl_id_2_
}

// CHECK-LABEL: test_global_id_dim
func.func @test_global_id_dim(%i: i32) -> index {
  %0 = sycl.global_id %i : index
  return %0 : index
}

// CHECK-LABEL: test_local_id
func.func @test_local_id() -> !sycl_id_3_ {
  %0 = sycl.local_id : !sycl_id_3_
  return %0 : !sycl_id_3_
}

// CHECK-LABEL: test_local_id_dim
func.func @test_local_id_dim(%i: i32) -> index {
  %0 = sycl.local_id %i : index
  return %0 : index
}

// CHECK-LABEL: test_global_offset
func.func @test_global_offset() -> !sycl_id_1_ {
  %0 = sycl.global_offset : !sycl_id_1_
  return %0 : !sycl_id_1_
}

// CHECK-LABEL: test_global_offset_dim
func.func @test_global_offset_dim(%i: i32) -> index {
  %0 = sycl.global_offset %i : index
  return %0 : index
}

// CHECK-LABEL: test_num_work_groups
func.func @test_num_work_groups() -> !sycl_range_2_ {
  %0 = sycl.num_work_groups : !sycl_range_2_
  return %0 : !sycl_range_2_
}

// CHECK-LABEL: test_num_work_groups_dim
func.func @test_num_work_groups_dim(%i: i32) -> index {
  %0 = sycl.num_work_groups %i : index
  return %0 : index
}

// CHECK-LABEL: test_work_group_size
func.func @test_work_group_size() -> !sycl_range_3_ {
  %0 = sycl.work_group_size : !sycl_range_3_
  return %0 : !sycl_range_3_
}

// CHECK-LABEL: test_work_group_size_dim
func.func @test_work_group_size_dim(%i: i32) -> index {
  %0 = sycl.work_group_size %i : index
  return %0 : index
}

// CHECK-LABEL: test_work_group_id
func.func @test_work_group_id() -> !sycl_id_1_ {
  %0 = sycl.work_group_id : !sycl_id_1_
  return %0 : !sycl_id_1_
}

// CHECK-LABEL: test_work_group_id_dim
func.func @test_work_group_id_dim(%i: i32) -> index {
  %0 = sycl.work_group_id %i : index
  return %0 : index
}

// CHECK-LABEL: test_num_sub_groups
func.func @test_num_sub_groups() -> i32 {
  %0 = sycl.num_sub_groups : i32
  return %0 : i32
}

// CHECK-LABEL: test_sub_group_max_size
func.func @test_sub_group_max_size() -> i32 {
  %0 = sycl.sub_group_max_size : i32
  return %0 : i32
}

// CHECK-LABEL: test_sub_group_size
func.func @test_sub_group_size() -> i32 {
  %0 = sycl.sub_group_size : i32
  return %0 : i32
}

// CHECK-LABEL: test_sub_group_id
func.func @test_sub_group_id() -> i32 {
  %0 = sycl.sub_group_id : i32
  return %0 : i32
}

// CHECK-LABEL: test_sub_group_local_id
func.func @test_sub_group_local_id() -> i32 {
  %0 = sycl.sub_group_local_id : i32
  return %0 : i32
}

// CHECK-LABEL: test_num_work_items_const
func.func @test_num_work_items_const() -> index {
  %c0_i32 = arith.constant 0 : i32
  %0 = sycl.num_work_items %c0_i32 : index
  return %0 : index
}

// CHECK-LABEL: test_global_id_const
func.func @test_global_id_const() -> index {
  %c1_i32 = arith.constant 1 : i32
  %0 = sycl.global_id %c1_i32 : index
  return %0 : index
}

// CHECK-LABEL: test_local_id_const
func.func @test_local_id_const() -> index {
  %c2_i32 = arith.constant 2 : i32
  %0 = sycl.local_id %c2_i32 : index
  return %0 : index
}

// CHECK-LABEL: test_global_offset_const
func.func @test_global_offset_const() -> index {
  %c0_i32 = arith.constant 0 : i32
  %0 = sycl.global_offset %c0_i32 : index
  return %0 : index
}

// CHECK-LABEL: test_num_work_groups_const
func.func @test_num_work_groups_const() -> index {
  %c1_i32 = arith.constant 1 : i32
  %0 = sycl.num_work_groups %c1_i32 : index
  return %0 : index
}

// CHECK-LABEL: test_work_group_size_const
func.func @test_work_group_size_const() -> index {
  %c2_i32 = arith.constant 2 : i32
  %0 = sycl.work_group_size %c2_i32 : index
  return %0 : index
}

// CHECK-LABEL: test_work_group_id_const
func.func @test_work_group_id_const() -> index {
  %c0_i32 = arith.constant 0 : i32
  %0 = sycl.work_group_id %c0_i32 : index
  return %0 : index
}

// CHECL-LABEL: test_accessor_get_pointer
func.func @test_accessor_get_pointer(%acc: memref<?x!sycl_accessor_1_i32_w_gb>) -> memref<?xi32, 1> {
  %0 = sycl.accessor.get_pointer(%acc) { ArgumentTypes = [memref<?x!sycl_accessor_1_i32_w_gb>], FunctionName = @"get_pointer", MangledFunctionName = @"get_pointer", TypeName = @"accessor" }  : (memref<?x!sycl_accessor_1_i32_w_gb>) -> memref<?xi32, 1>
  return %0 : memref<?xi32, 1>
}

// CHECL-LABEL: test_accessor_get_range
func.func @test_accessor_get_range(%acc: memref<?x!sycl_accessor_1_i32_w_gb>) -> !sycl_range_1_ {
  %0 = sycl.accessor.get_range(%acc) { ArgumentTypes = [memref<?x!sycl_accessor_1_i32_w_gb>], FunctionName = @"get_range", MangledFunctionName = @"get_range", TypeName = @"accessor" }  : (memref<?x!sycl_accessor_1_i32_w_gb>) -> !sycl_range_1_
  return %0 : !sycl_range_1_
}

// CHECK-LABEL: test_accessor_subscript_atomic
func.func @test_accessor_subscript_atomic(
  %acc: memref<?x!sycl_accessor_1_i32_ato_gb>, 
  %idx: memref<?x!sycl_id_1_>) -> !sycl_atomic_i32_1_ {
  %0 = sycl.accessor.subscript %acc[%idx] { 
        ArgumentTypes = [memref<?x!sycl_accessor_1_i32_ato_gb>, 
                          memref<?x!sycl_id_1_>], 
        FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @"id" }  : (memref<?x!sycl_accessor_1_i32_ato_gb>, 
                                memref<?x!sycl_id_1_>) -> !sycl_atomic_i32_1_
  return %0 : !sycl_atomic_i32_1_
}
