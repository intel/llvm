// RUN: polygeist-opt --loop-internalization --split-input-file -mlir-pass-statistics -o /dev/null %s 2>&1 | FileCheck %s

// COM: Not candidate: kernel uses local_accessor
// CHECK: LoopInternalization
// CHECK-DAG:   (S) 0 num-access-internalized - Number of accesses internalized
// CHECK-DAG:   (S) 0 num-loop-internalized - Number of loops internalized

!sycl_array_2 = !sycl.array<[2], (memref<2xi64, 4>)>
!sycl_id_2 = !sycl.id<[2], (!sycl_array_2)>
!sycl_range_2 = !sycl.range<[2], (!sycl_array_2)>
!sycl_accessor_impl_device_2 = !sycl.accessor_impl_device<[2], (!sycl_id_2, !sycl_range_2, !sycl_range_2)>
!sycl_group_2 = !sycl.group<[2], (!sycl_range_2, !sycl_range_2, !sycl_range_2, !sycl_id_2)>
!sycl_item_base_2 = !sycl.item_base<[2, true], (!sycl_range_2, !sycl_id_2, !sycl_id_2)>
!sycl_accessor_2_f32_r_dev = !sycl.accessor<[2, f32, read, device], (!sycl_accessor_impl_device_2, !llvm.struct<(memref<?xf32, 2>)>)>
!sycl_item_2 = !sycl.item<[2, true], (!sycl_item_base_2)>
!sycl_nd_item_2 = !sycl.nd_item<[2], (!sycl_item_2, !sycl_item_2, !sycl_group_2)>

gpu.module @device_func {
func.func private @callee1(%arg0: memref<?x!sycl_accessor_2_f32_r_dev>, %arg1: memref<?x!sycl_nd_item_2>) {
  %alloca = memref.alloca() : memref<1x!sycl_id_2>
  %id = memref.cast %alloca : memref<1x!sycl_id_2> to memref<?x!sycl_id_2>
  %c0_i32 = arith.constant 0 : i32
  %tx = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_2>, i32) -> i64

  affine.for %ii = 0 to 256 {
    %i = arith.index_cast %ii : index to i64
    sycl.constructor @id(%id, %tx, %i) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2>, i64, i64)
    %subscr1 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<?x!sycl_id_2>) -> memref<?xf32>
    %load1 = affine.load %subscr1[0] : memref<?xf32>
  }
  return
}
gpu.func @caller1(%arg0: memref<?xi32, #sycl.access.address_space<local>>, %arg1: memref<?x!sycl_accessor_2_f32_r_dev>, %arg2: memref<?x!sycl_nd_item_2>) kernel {
  func.call @callee1(%arg1, %arg2) : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<?x!sycl_nd_item_2>) -> ()
  gpu.return
}

func.func private @callee2(%arg0: memref<?x!sycl_accessor_2_f32_r_dev>, %arg1: memref<?x!sycl_nd_item_2>) {
  %alloca = memref.alloca() : memref<1x!sycl_id_2>
  %id = memref.cast %alloca : memref<1x!sycl_id_2> to memref<?x!sycl_id_2>
  %c0_i32 = arith.constant 0 : i32
  %tx = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_2>, i32) -> i64

  affine.for %ii = 0 to 256 {
    %i = arith.index_cast %ii : index to i64
    sycl.constructor @id(%id, %tx, %i) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2>, i64, i64)
    %subscr1 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<?x!sycl_id_2>) -> memref<?xf32>
    %load1 = affine.load %subscr1[0] : memref<?xf32>
  }
  return
}
// COM: Not candidate: kernel uses local_accessor
gpu.func @caller2(%arg0: memref<?xi32, 3>, %arg1: memref<?x!sycl_accessor_2_f32_r_dev>, %arg2: memref<?x!sycl_nd_item_2>) kernel {
  func.call @callee2(%arg1, %arg2) : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<?x!sycl_nd_item_2>) -> ()
  gpu.return
}
}

// -----

// COM: Not candidate: work group sizes do not match
// CHECK: LoopInternalization
// CHECK-DAG:   (S) 0 num-access-internalized - Number of accesses internalized
// CHECK-DAG:   (S) 0 num-loop-internalized - Number of loops internalized

!sycl_array_2 = !sycl.array<[2], (memref<2xi64, 4>)>
!sycl_id_2 = !sycl.id<[2], (!sycl_array_2)>
!sycl_range_2 = !sycl.range<[2], (!sycl_array_2)>
!sycl_accessor_impl_device_2 = !sycl.accessor_impl_device<[2], (!sycl_id_2, !sycl_range_2, !sycl_range_2)>
!sycl_group_2 = !sycl.group<[2], (!sycl_range_2, !sycl_range_2, !sycl_range_2, !sycl_id_2)>
!sycl_item_base_2 = !sycl.item_base<[2, true], (!sycl_range_2, !sycl_id_2, !sycl_id_2)>
!sycl_accessor_2_f32_r_dev = !sycl.accessor<[2, f32, read, device], (!sycl_accessor_impl_device_2, !llvm.struct<(memref<?xf32, 2>)>)>
!sycl_item_2 = !sycl.item<[2, true], (!sycl_item_base_2)>
!sycl_nd_item_2 = !sycl.nd_item<[2], (!sycl_item_2, !sycl_item_2, !sycl_group_2)>

gpu.module @device_func {
func.func private @affine_2d(%arg0: memref<?x!sycl_accessor_2_f32_r_dev>, %arg1: memref<?x!sycl_nd_item_2>) {
  %alloca = memref.alloca() : memref<1x!sycl_id_2>
  %id = memref.cast %alloca : memref<1x!sycl_id_2> to memref<?x!sycl_id_2>
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %tx = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_2>, i32) -> i64
  %ty = sycl.nd_item.get_global_id(%arg1, %c1_i32) : (memref<?x!sycl_nd_item_2>, i32) -> i64

  affine.for %ii = 0 to 256 {
    %i = arith.index_cast %ii : index to i64
    sycl.constructor @id(%id, %tx, %i) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2>, i64, i64)
    %subscr1 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<?x!sycl_id_2>) -> memref<?xf32>
    %load1 = affine.load %subscr1[0] : memref<?xf32>

    sycl.constructor @id(%id, %i, %ty) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2>, i64, i64)    
    %subscr2 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<?x!sycl_id_2>) -> memref<?xf32>
    %load2 = affine.load %subscr2[0] : memref<?xf32>
  }
  return
}
gpu.func @kernel(%arg0: memref<?x!sycl_accessor_2_f32_r_dev>, %arg1: memref<?x!sycl_nd_item_2>) kernel attributes {reqd_work_group_size = [4, 2]} {
  func.call @affine_2d(%arg0, %arg1) : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<?x!sycl_nd_item_2>) -> ()
  gpu.return
}
}

// -----

// COM: Not candidate: Access references more than one global id or innermost loop induction variable.
// CHECK: LoopInternalization
// CHECK-DAG:   (S) 0 num-access-internalized - Number of accesses internalized
// CHECK-DAG:   (S) 0 num-loop-internalized - Number of loops internalized

!sycl_array_2 = !sycl.array<[2], (memref<2xi64, 4>)>
!sycl_id_2 = !sycl.id<[2], (!sycl_array_2)>
!sycl_range_2 = !sycl.range<[2], (!sycl_array_2)>
!sycl_accessor_impl_device_2 = !sycl.accessor_impl_device<[2], (!sycl_id_2, !sycl_range_2, !sycl_range_2)>
!sycl_group_2 = !sycl.group<[2], (!sycl_range_2, !sycl_range_2, !sycl_range_2, !sycl_id_2)>
!sycl_item_base_2 = !sycl.item_base<[2, true], (!sycl_range_2, !sycl_id_2, !sycl_id_2)>
!sycl_accessor_2_f32_r_dev = !sycl.accessor<[2, f32, read, device], (!sycl_accessor_impl_device_2, !llvm.struct<(memref<?xf32, 2>)>)>
!sycl_item_2 = !sycl.item<[2, true], (!sycl_item_base_2)>
!sycl_nd_item_2 = !sycl.nd_item<[2], (!sycl_item_2, !sycl_item_2, !sycl_group_2)>

gpu.module @device_func {
func.func private @affine_2d(%arg0: memref<?x!sycl_accessor_2_f32_r_dev>, %arg1: memref<?x!sycl_nd_item_2>) {
  %alloca = memref.alloca() : memref<1x!sycl_id_2>
  %id = memref.cast %alloca : memref<1x!sycl_id_2> to memref<?x!sycl_id_2>
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %tx = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_2>, i32) -> i64
  %ty = sycl.nd_item.get_global_id(%arg1, %c1_i32) : (memref<?x!sycl_nd_item_2>, i32) -> i64

  affine.for %ii = 0 to 256 {
    %i = arith.index_cast %ii : index to i64
    %add1 = arith.addi %i, %ty : i64
    sycl.constructor @id(%id, %tx, %add1) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2>, i64, i64)
    %subscr1 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<?x!sycl_id_2>) -> memref<?xf32>
    %load1 = affine.load %subscr1[0] : memref<?xf32>

    %add2 = arith.addi %tx, %ty : i64
    sycl.constructor @id(%id, %add2, %i) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2>, i64, i64)    
    %subscr2 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<?x!sycl_id_2>) -> memref<?xf32>
    %load2 = affine.load %subscr2[0] : memref<?xf32>
  }
  return
}
gpu.func @kernel(%arg0: memref<?x!sycl_accessor_2_f32_r_dev>, %arg1: memref<?x!sycl_nd_item_2>) kernel attributes {reqd_work_group_size = [4, 2]} {
  func.call @affine_2d(%arg0, %arg1) : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<?x!sycl_nd_item_2>) -> ()
  gpu.return
}
}

// -----

// COM: Not candidate: innermost loop is not uniform, i.e., not all threads execute the innermost loop.
// CHECK: LoopInternalization
// CHECK-DAG:   (S) 0 num-access-internalized - Number of accesses internalized
// CHECK-DAG:   (S) 0 num-loop-internalized - Number of loops internalized

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_item_base_1_ = !sycl.item_base<[2, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_accessor_1_f32_r_dev = !sycl.accessor<[1, f32, read, device], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xf32, 2>)>)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>

gpu.module @device_func {
func.func private @affine(%alloca_cond: memref<1xi1>, %arg0: memref<?x!sycl_accessor_1_f32_r_dev>, %arg1: memref<?x!sycl_item_1_>) {
  %alloca = memref.alloca() : memref<1x!sycl_id_1_>
  %id = memref.cast %alloca : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
  %c0_i32 = arith.constant 0 : i32
  %tx = sycl.item.get_id(%arg1, %c0_i32) : (memref<?x!sycl_item_1_>, i32) -> i64

 // condition is non-uniform (intra-procedurally).
  %c0_i64 = arith.constant 0 : i64 
  %cond = arith.cmpi sgt, %tx, %c0_i64 : i64
  scf.if %cond {
    affine.for %ii = 0 to 256 {
      %i = arith.index_cast %ii : index to i64
      sycl.constructor @id(%id, %i) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_1_>, i64)
      %subscr1 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_1_f32_r_dev>, memref<?x!sycl_id_1_>) -> memref<?xf32>
      %load1 = affine.load %subscr1[0] : memref<?xf32>
    }
  }

  // condition is non-uniform (inter-procedurally).
  %c0_index = arith.constant 0 : index
  %cond1 = memref.load %alloca_cond[%c0_index] : memref<1xi1>
  scf.if %cond1 {
    affine.for %ii = 0 to 256 {
      %i = arith.index_cast %ii : index to i64
      sycl.constructor @id(%id, %i) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_1_>, i64)
      %subscr1 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_1_f32_r_dev>, memref<?x!sycl_id_1_>) -> memref<?xf32>
      %load1 = affine.load %subscr1[0] : memref<?xf32>
    }
  }

  return
}

gpu.func @kernel(%arg0: memref<?x!sycl_accessor_1_f32_r_dev>, %arg1: memref<?x!sycl_item_1_>) kernel {
  %alloca = memref.alloca() : memref<1xi1>
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c0_i64 = arith.constant 0 : i64  
  %tx = sycl.item.get_id(%arg1, %c0_i32) : (memref<?x!sycl_item_1_>, i32) -> i64  
  %cond = arith.cmpi sgt, %tx, %c0_i64 : i64

  // COM: Store the condition (non-uniform) into memory.
  memref.store %cond, %alloca[%c0] : memref<1xi1>

  func.call @affine(%alloca, %arg0, %arg1) : (memref<1xi1>, memref<?x!sycl_accessor_1_f32_r_dev>, memref<?x!sycl_item_1_>) -> ()
  gpu.return
}
}
