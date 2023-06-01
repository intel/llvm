// RUN: polygeist-opt --loop-internalization --split-input-file -allow-unregistered-dialect %s | FileCheck %s --check-prefixes=CHECK,SIZE1
// RUN: polygeist-opt --loop-internalization --loop-internalization-tile-size=2 --split-input-file -allow-unregistered-dialect %s | FileCheck %s --check-prefixes=CHECK,SIZE2

!sycl_array_2 = !sycl.array<[2], (memref<2xi64, 4>)>
!sycl_id_2 = !sycl.id<[2], (!sycl_array_2)>
!sycl_range_2 = !sycl.range<[2], (!sycl_array_2)>
!sycl_accessor_impl_device_2 = !sycl.accessor_impl_device<[2], (!sycl_id_2, !sycl_range_2, !sycl_range_2)>
!sycl_group_2 = !sycl.group<[2], (!sycl_range_2, !sycl_range_2, !sycl_range_2, !sycl_id_2)>
!sycl_item_base_2 = !sycl.item_base<[2, true], (!sycl_range_2, !sycl_id_2, !sycl_id_2)>
!sycl_accessor_2_f32_r_gb = !sycl.accessor<[2, f32, read, global_buffer], (!sycl_accessor_impl_device_2, !llvm.struct<(memref<?xf32, 2>)>)>
!sycl_item_2 = !sycl.item<[2, true], (!sycl_item_base_2)>
!sycl_nd_item_2 = !sycl.nd_item<[2], (!sycl_item_2, !sycl_item_2, !sycl_group_2)>

// CHECK-DAG:   [[MAP1:#map.*]] = affine_map<()[s0] -> (256 ceildiv s0)>
// CHECK-DAG:   [[MAP2:#map.*]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG:   [[MAP3:#map.*]] = affine_map<(d0)[s0] -> (d0 * s0 + s0, 256)>
// CHECK-LABEL: func.func private @affine_2d(%arg0: memref<?x!sycl_accessor_2_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_2_>) {
// CHECK:         %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    [[TX:%.*]] = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_2_>, i32) -> i64  
// SIZE1-NEXT:    [[TILESIZE:%.*]] = arith.constant 1 : index
// SIZE2-NEXT:    [[TILESIZE:%.*]] = arith.constant 2 : index
// CHECK-NEXT:    affine.for [[IV1:%.*]] = 0 to [[MAP1]]()[[[TILESIZE]]] {
// CHECK-NEXT:      spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:      affine.for [[IV2:%.*]] = [[MAP2]]([[IV1]])[[[TILESIZE]]] to min [[MAP3]]([[IV1]])[[[TILESIZE]]] {
// CHECK-NEXT:        [[IV2_CAST:%.*]] = arith.index_cast [[IV2]] : index to i64 
// CHECK-NEXT:        sycl.constructor @id([[ID:%.*]], [[TX]], [[IV2_CAST]]) {{.*}} : (memref<?x!sycl_id_2_>, i64, i64)
// CHECK-NEXT:        [[SUBSCR:%.*]] = sycl.accessor.subscript %arg0[[[ID]]] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<?x!sycl_id_2_>) -> memref<?xf32>
// CHECK-NEXT:        {{.*}} = affine.load [[SUBSCR]][0] : memref<?xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
gpu.module @device_func {
func.func private @affine_2d(%arg0: memref<?x!sycl_accessor_2_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_2>) {
  %alloca = memref.alloca() : memref<1x!sycl_id_2>
  %id = memref.cast %alloca : memref<1x!sycl_id_2> to memref<?x!sycl_id_2>
  %c0_i32 = arith.constant 0 : i32
  %tx = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_2>, i32) -> i64

  affine.for %ii = 0 to 256 {
    %i = arith.index_cast %ii : index to i64
    sycl.constructor @id(%id, %tx, %i) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2>, i64, i64)
    %subscr1 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<?x!sycl_id_2>) -> memref<?xf32>
    %load1 = affine.load %subscr1[0] : memref<?xf32>
  }
  return
}
gpu.func @kernel(%arg0: memref<?x!sycl_accessor_2_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_2>) kernel {
  func.call @affine_2d(%arg0, %arg1) : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<?x!sycl_nd_item_2>) -> ()
  gpu.return
}
}

// -----
!sycl_array_3 = !sycl.array<[3], (memref<3xi64, 4>)>
!sycl_id_3 = !sycl.id<[3], (!sycl_array_3)>
!sycl_range_3 = !sycl.range<[3], (!sycl_array_3)>
!sycl_accessor_impl_device_3 = !sycl.accessor_impl_device<[3], (!sycl_id_3, !sycl_range_3, !sycl_range_3)>
!sycl_group_3 = !sycl.group<[3], (!sycl_range_3, !sycl_range_3, !sycl_range_3, !sycl_id_3)>
!sycl_item_base_3 = !sycl.item_base<[3, true], (!sycl_range_3, !sycl_id_3, !sycl_id_3)>
!sycl_accessor_3_f32_r_gb = !sycl.accessor<[3, f32, read, global_buffer], (!sycl_accessor_impl_device_3, !llvm.struct<(memref<?xf32, 3>)>)>
!sycl_item_3 = !sycl.item<[3, true], (!sycl_item_base_3)>
!sycl_nd_item_3 = !sycl.nd_item<[3], (!sycl_item_3, !sycl_item_3, !sycl_group_3)>

// CHECK-DAG:   [[MAP1:#map.*]] = affine_map<()[s0] -> (511 ceildiv s0 + 1)>
// CHECK-DAG:   [[MAP2:#map.*]] = affine_map<(d0)[s0] -> ((d0 - 1) * s0 + 1)>
// CHECK-DAG:   [[MAP3:#map.*]] = affine_map<(d0)[s0] -> ((d0 - 1) * s0 + s0 + 1, 512)>
// CHECK-LABEL: func.func private @affine_3d(%arg0: memref<?x!sycl_accessor_3_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_3_>) {
// CHECK:         %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    [[TX:%.*]] = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_3_>, i32) -> i64  
// CHECK-NEXT:    affine.for [[IV1:%.*]] = 0 to 256 {
// SIZE1-NEXT:      [[TILESIZE:%.*]] = arith.constant 1 : index
// SIZE2-NEXT:      [[TILESIZE:%.*]] = arith.constant 2 : index
// CHECK-NEXT:      affine.for [[IV2:%.*]] = 1 to [[MAP1]]()[[[TILESIZE]]] {
// CHECK-NEXT:        spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:        affine.for [[IV3:%.*]] = [[MAP2]]([[IV2]])[[[TILESIZE]]] to min [[MAP3]]([[IV2]])[[[TILESIZE]]] {
// CHECK-DAG:           [[IV1_CAST:%.*]] = arith.index_cast [[IV1]] : index to i64   
// CHECK-DAG:           [[IV3_CAST:%.*]] = arith.index_cast [[IV3]] : index to i64 
// CHECK-NEXT:          sycl.constructor @id([[ID:%.*]], [[TX]], [[IV1_CAST]], [[IV3_CAST]]) {{.*}} : (memref<?x!sycl_id_3_>, i64, i64, i64)
// CHECK-NEXT:          [[SUBSCR:%.*]] = sycl.accessor.subscript %arg0[[[ID]]] : (memref<?x!sycl_accessor_3_f32_r_gb>, memref<?x!sycl_id_3_>) -> memref<?xf32>
// CHECK-NEXT:          {{.*}} = affine.load [[SUBSCR]][0] : memref<?xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
gpu.module @device_func {
func.func private @affine_3d(%arg0: memref<?x!sycl_accessor_3_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_3>) {
  %alloca = memref.alloca() : memref<1x!sycl_id_3>
  %id = memref.cast %alloca : memref<1x!sycl_id_3> to memref<?x!sycl_id_3>
  %c0_i32 = arith.constant 0 : i32
  %tx = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_3>, i32) -> i64

  affine.for %ii = 0 to 256 {
    affine.for %jj = 1 to 512 {
      %i = arith.index_cast %ii : index to i64
      %j = arith.index_cast %jj : index to i64
      sycl.constructor @id(%id, %tx, %i, %j) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_3>, i64, i64, i64)
      %subscr1 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_3_f32_r_gb>, memref<?x!sycl_id_3>) -> memref<?xf32>
      %load1 = affine.load %subscr1[0] : memref<?xf32>
    }
  }
  return
}
gpu.func @kernel(%arg0: memref<?x!sycl_accessor_3_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_3>) kernel {
  func.call @affine_3d(%arg0, %arg1) : (memref<?x!sycl_accessor_3_f32_r_gb>, memref<?x!sycl_nd_item_3>) -> ()
  gpu.return
}
}

// -----
!sycl_array_2 = !sycl.array<[2], (memref<2xi64, 4>)>
!sycl_id_2 = !sycl.id<[2], (!sycl_array_2)>
!sycl_range_2 = !sycl.range<[2], (!sycl_array_2)>
!sycl_accessor_impl_device_2 = !sycl.accessor_impl_device<[2], (!sycl_id_2, !sycl_range_2, !sycl_range_2)>
!sycl_group_2 = !sycl.group<[2], (!sycl_range_2, !sycl_range_2, !sycl_range_2, !sycl_id_2)>
!sycl_item_base_2 = !sycl.item_base<[2, true], (!sycl_range_2, !sycl_id_2, !sycl_id_2)>
!sycl_accessor_2_f32_r_gb = !sycl.accessor<[2, f32, read, global_buffer], (!sycl_accessor_impl_device_2, !llvm.struct<(memref<?xf32, 2>)>)>
!sycl_item_2 = !sycl.item<[2, true], (!sycl_item_base_2)>
!sycl_nd_item_2 = !sycl.nd_item<[2], (!sycl_item_2, !sycl_item_2, !sycl_group_2)>

// CHECK-LABEL: func.func private @scf_2d(%arg0: memref<?x!sycl_accessor_2_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_2_>) {
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c256 = arith.constant 256 : index
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    [[TX:%.*]] = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_2_>, i32) -> i64  
// SIZE1-NEXT:    [[TILESIZE:%.*]] = arith.constant 1 : index
// SIZE2-NEXT:    [[TILESIZE:%.*]] = arith.constant 2 : index
// CHECK-NEXT:    [[STEP:%.*]] = arith.muli %c1, [[TILESIZE]] : index
// CHECK-NEXT:    scf.for [[IV1:%.*]] = %c0 to %c256 step [[STEP]] {
// CHECK-NEXT:      spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:      [[VAL_1:%.*]] = arith.addi [[IV1]], [[STEP]] : index
// CHECK-NEXT:      [[VAL_2:%.*]] = arith.cmpi slt, %c256, [[VAL_1]] : index
// CHECK-NEXT:      [[VAL_3:%.*]] = arith.select [[VAL_2]], %c256, [[VAL_1]] : index
// CHECK-NEXT:      scf.for [[IV2:%.*]] = [[IV1]] to [[VAL_3]] step %c1 {
// CHECK-NEXT:        [[IV2_CAST:%.*]] = arith.index_cast [[IV2]] : index to i64   
// CHECK-NEXT:        sycl.constructor @id([[ID:%.*]], [[TX]], [[IV2_CAST]]) {{.*}} : (memref<?x!sycl_id_2_>, i64, i64)
// CHECK-NEXT:        [[SUBSCR:%.*]] = sycl.accessor.subscript %arg0[[[ID]]] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<?x!sycl_id_2_>) -> memref<?xf32>
// CHECK-NEXT:        {{.*}} = affine.load [[SUBSCR]][0] : memref<?xf32>  
// CHECK-NEXT:      }
// CHECK-NEXT:      spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
gpu.module @device_func {
func.func private @scf_2d(%arg0: memref<?x!sycl_accessor_2_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_2>) {
  %alloca = memref.alloca() : memref<1x!sycl_id_2>
  %id = memref.cast %alloca : memref<1x!sycl_id_2> to memref<?x!sycl_id_2>
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32  
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %tx = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_2>, i32) -> i64

  scf.for %ii = %c0 to %c256 step %c1 {
    %i = arith.index_cast %ii : index to i64    
    sycl.constructor @id(%id, %tx, %i) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2>, i64, i64)
    %subscr1 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<?x!sycl_id_2>) -> memref<?xf32>
    %load1 = affine.load %subscr1[0] : memref<?xf32>
  }
  return
}
gpu.func @kernel(%arg0: memref<?x!sycl_accessor_2_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_2>) kernel {
  func.call @scf_2d(%arg0, %arg1) : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<?x!sycl_nd_item_2>) -> ()
  gpu.return
}
}

// -----
!sycl_array_3 = !sycl.array<[3], (memref<3xi64, 4>)>
!sycl_id_3 = !sycl.id<[3], (!sycl_array_3)>
!sycl_range_3 = !sycl.range<[3], (!sycl_array_3)>
!sycl_accessor_impl_device_3 = !sycl.accessor_impl_device<[3], (!sycl_id_3, !sycl_range_3, !sycl_range_3)>
!sycl_group_3 = !sycl.group<[3], (!sycl_range_3, !sycl_range_3, !sycl_range_3, !sycl_id_3)>
!sycl_item_base_3 = !sycl.item_base<[3, true], (!sycl_range_3, !sycl_id_3, !sycl_id_3)>
!sycl_accessor_3_f32_r_gb = !sycl.accessor<[3, f32, read, global_buffer], (!sycl_accessor_impl_device_3, !llvm.struct<(memref<?xf32, 3>)>)>
!sycl_item_3 = !sycl.item<[3, true], (!sycl_item_base_3)>
!sycl_nd_item_3 = !sycl.nd_item<[3], (!sycl_item_3, !sycl_item_3, !sycl_group_3)>

// CHECK-LABEL: func.func private @scf_3d(%arg0: memref<?x!sycl_accessor_3_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_3_>) {
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c256 = arith.constant 256 : index
// CHECK-DAG:     %c512 = arith.constant 512 : index
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    [[TX:%.*]] = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_3_>, i32) -> i64  
// CHECK-NEXT:    scf.for [[IV1:.*]] = %c0 to %c256 step %c1 {
// SIZE1-NEXT:      [[TILESIZE:%.*]] = arith.constant 1 : index
// SIZE2-NEXT:      [[TILESIZE:%.*]] = arith.constant 2 : index
// CHECK-NEXT:      [[STEP:%.*]] = arith.muli %c1, [[TILESIZE]] : index
// CHECK-NEXT:      scf.for [[IV2:.*]] = %c1 to %c512 step [[STEP]] {
// CHECK-NEXT:        spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:        [[VAL_2:%.*]] = arith.addi [[IV2]], [[STEP]] : index
// CHECK-NEXT:        [[VAL_6:%.*]] = arith.cmpi slt, %c512, [[VAL_2]] : index
// CHECK-NEXT:        [[VAL_7:%.*]] = arith.select [[VAL_6]], %c512, [[VAL_2]] : index
// CHECK-NEXT:        scf.for [[IV4:.*]] = [[IV2]] to [[VAL_7]] step %c1 {
// CHECK-DAG:           [[IV1_CAST:%.*]] = arith.index_cast [[IV1]] : index to i64   
// CHECK-DAG:           [[IV4_CAST:%.*]] = arith.index_cast [[IV4]] : index to i64 
// CHECK-NEXT:          sycl.constructor @id([[ID:%.*]], [[TX]], [[IV1_CAST]], [[IV4_CAST]]) {{.*}} : (memref<?x!sycl_id_3_>, i64, i64, i64)
// CHECK-NEXT:          [[SUBSCR:%.*]] = sycl.accessor.subscript %arg0[[[ID]]] : (memref<?x!sycl_accessor_3_f32_r_gb>, memref<?x!sycl_id_3_>) -> memref<?xf32>
// CHECK-NEXT:          {{.*}} = affine.load [[SUBSCR]][0] : memref<?xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
gpu.module @device_func {
func.func private @scf_3d(%arg0: memref<?x!sycl_accessor_3_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_3>) {
  %alloca = memref.alloca() : memref<1x!sycl_id_3>
  %id = memref.cast %alloca : memref<1x!sycl_id_3> to memref<?x!sycl_id_3>
  %c0_i32 = arith.constant 0 : i32  
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %tx = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_3>, i32) -> i64

  scf.for %ii = %c0 to %c256 step %c1 {
    scf.for %jj = %c1 to %c512 step %c1 {
      %i = arith.index_cast %ii : index to i64
      %j = arith.index_cast %jj : index to i64
      sycl.constructor @id(%id, %tx, %i, %j) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_3>, i64, i64, i64)
      %subscr1 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_3_f32_r_gb>, memref<?x!sycl_id_3>) -> memref<?xf32>
      %load1 = affine.load %subscr1[0] : memref<?xf32>
    }
  }
  return
}
gpu.func @kernel(%arg0: memref<?x!sycl_accessor_3_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_3>) kernel {
  func.call @scf_3d(%arg0, %arg1) : (memref<?x!sycl_accessor_3_f32_r_gb>, memref<?x!sycl_nd_item_3>) -> ()
  gpu.return
}
}
