// RUN: polygeist-opt --loop-internalization --split-input-file -allow-unregistered-dialect %s | FileCheck %s --check-prefixes=CHECK,SIZE1
// RUN: polygeist-opt --loop-internalization="tile-size=2" --split-input-file -allow-unregistered-dialect %s | FileCheck %s --check-prefixes=CHECK,SIZE2

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
// CHECK:       memref.global "private" @WGLocalMem : memref<64xi8, #sycl.access.address_space<local>>
// CHECK-LABEL: func.func private @affine_2d(%arg0: memref<?x!sycl_accessor_2_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_2_>) {

// COM: Get local ids:
// CHECK:         [[LOCALID:%.*]] = sycl.local_id : !sycl_id_2_1
// CHECK-NEXT:    [[LOCALID_:%.*]] = memref.alloca() : memref<1x!sycl_id_2_1>
// CHECK:         memref.store [[LOCALID]], [[LOCALID_:%.*]][%c0{{.*}}] : memref<1x!sycl_id_2_1>
// CHECK:         [[LOCALID0_:%.*]] = sycl.id.get [[LOCALID_]][%c0{{.*}}] : (memref<1x!sycl_id_2_1>, i32) -> memref<?xindex>
// CHECK-NEXT:    [[LOCALID0:%.*]] = memref.load [[LOCALID0_]][%c0{{.*}}] : memref<?xindex>
// CHECK:         [[LOCALID1_:%.*]] = sycl.id.get [[LOCALID_]][%c1{{.*}}] : (memref<1x!sycl_id_2_1>, i32) -> memref<?xindex>
// CHECK-NEXT:    [[LOCALID1:%.*]] = memref.load [[LOCALID1_]][%c0{{.*}}] : memref<?xindex>

// CHECK:         [[TX:%.*]] = sycl.nd_item.get_global_id(%arg1, %c0{{.*}}) : (memref<?x!sycl_nd_item_2_>, i32) -> i64  

// COM: Get pointer to local memory:
// CHECK-NEXT:    [[GETGLOBAL:%.*]] = memref.get_global @WGLocalMem : memref<64xi8, #sycl.access.address_space<local>>

// SIZE1-NEXT:    [[TILESIZE:%.*]] = arith.constant 1 : index
// SIZE2-NEXT:    [[TILESIZE:%.*]] = arith.constant 2 : index
// CHECK-NEXT:    affine.for [[IV1:%.*]] = 0 to [[MAP1]]()[[[TILESIZE]]] {

// COM: Get pointer to memref for 1st load:
// CHECK:           [[ID0:%.*]] = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK:           [[ID0GET0:%.*]] = sycl.id.get [[ID0]][%c0{{.*}}] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
// CHECK:           memref.store [[LOCALID0]], [[ID0GET0]][%c0{{.*}}] : memref<?xindex>
// CHECK:           [[ID0GET1:%.*]] = sycl.id.get [[ID0]][%c1{{.*}}] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
// CHECK:           memref.store [[LOCALID1]], [[ID0GET1]][%c0{{.*}}] : memref<?xindex>
// CHECK:           [[ACCSUB0:%.*]] = sycl.accessor.subscript %arg0[[[ID0]]] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<1x!sycl_id_2_>) -> memref<?xf32, 1>

// COM: Get pointer to local memory for 1st load:
// CHECK:           [[VIEW0:%.*]] = memref.view [[GETGLOBAL]][%c0{{.*}}][] : memref<64xi8, #sycl.access.address_space<local>> to memref<4x2xf32, #sycl.access.address_space<local>>

// COM: Get pointer to memref for 2nd load:
// CHECK:           [[ID1:%.*]] = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK:           [[ID1GET0:%.*]] = sycl.id.get [[ID1]][%c0{{.*}}] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
// CHECK:           memref.store [[LOCALID0]], [[ID1GET0]][%c0{{.*}}] : memref<?xindex>
// CHECK:           [[ID1GET1:%.*]] = sycl.id.get [[ID1]][%c1{{.*}}] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
// CHECK:           memref.store [[LOCALID1]], [[ID1GET1]][%c0{{.*}}] : memref<?xindex>
// CHECK:           [[ACCSUB1:%.*]] = sycl.accessor.subscript %arg0[[[ID1]]] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<1x!sycl_id_2_>) -> memref<?xf32, 1>

// COM: Get pointer to local memory for 2nd load:
// CHECK:           [[VIEW1:%.*]] = memref.view [[GETGLOBAL]][%c32][] : memref<64xi8, #sycl.access.address_space<local>> to memref<4x2xf32, #sycl.access.address_space<local>>

// CHECK-NEXT:      spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:      affine.for [[IV2:%.*]] = [[MAP2]]([[IV1]])[[[TILESIZE]]] to min [[MAP3]]([[IV1]])[[[TILESIZE]]] {
// CHECK-NEXT:        [[IV2_CAST:%.*]] = arith.index_cast [[IV2]] : index to i64 
// CHECK-NEXT:        sycl.constructor @id([[ID:%.*]], [[TX]], [[IV2_CAST]]) {{.*}} : (memref<?x!sycl_id_2_>, i64, i64)
// CHECK-NEXT:        [[SUBSCR1:%.*]] = sycl.accessor.subscript %arg0[[[ID]]] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<?x!sycl_id_2_>) -> memref<?xf32>
// CHECK-NEXT:        {{.*}} = affine.load [[SUBSCR1]][0] : memref<?xf32>
// CHECK-NEXT:        [[SUBSCR2:%.*]] = sycl.accessor.subscript %arg0[[[ID]]] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<?x!sycl_id_2_>) -> memref<?xf32>
// CHECK-NEXT:        {{.*}} = affine.load [[SUBSCR2]][0] : memref<?xf32>
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
  %c1_i32 = arith.constant 1 : i32
  %tx = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_2>, i32) -> i64

  affine.for %ii = 0 to 256 {
    %i = arith.index_cast %ii : index to i64
    sycl.constructor @id(%id, %tx, %i) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2>, i64, i64)
    %subscr1 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<?x!sycl_id_2>) -> memref<?xf32>
    %load1 = affine.load %subscr1[0] : memref<?xf32>
    %subscr2 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<?x!sycl_id_2>) -> memref<?xf32>
    %load2 = affine.load %subscr2[0] : memref<?xf32>
  }
  return
}
gpu.func @kernel(%arg0: memref<?x!sycl_accessor_2_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_2>) kernel attributes {reqd_work_group_size = [4, 2]} {
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
// CHECK:       memref.global "private" @WGLocalMem : memref<32000xi8, #sycl.access.address_space<local>>
// CHECK-LABEL: func.func private @affine_3d(%arg0: memref<?x!sycl_accessor_3_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_3_>) {

// COM: Get work group sizes:
// CHECK:         [[LOCALRANGE0:%.*]] = sycl.nd_item.get_local_range(%arg1, %c0{{.*}}) : (memref<?x!sycl_nd_item_3_>, i32) -> i64 
// CHECK-NEXT:    [[LOCALRANGE0_:%.*]] = arith.index_cast [[LOCALRANGE0]] : i64 to index 
// CHECK:         [[LOCALRANGE1:%.*]] = sycl.nd_item.get_local_range(%arg1, %c1{{.*}}) : (memref<?x!sycl_nd_item_3_>, i32) -> i64 
// CHECK-NEXT:    [[LOCALRANGE1_:%.*]] = arith.index_cast [[LOCALRANGE1]] : i64 to index
// CHECK:         [[LOCALRANGE2:%.*]] = sycl.nd_item.get_local_range(%arg1, %c2{{.*}}) : (memref<?x!sycl_nd_item_3_>, i32) -> i64 
// CHECK-NEXT:    [[LOCALRANGE2_:%.*]] = arith.index_cast [[LOCALRANGE2]] : i64 to index 

// COM: Get local ids:
// CHECK:         [[LOCALID:%.*]] = sycl.local_id : !sycl_id_3_1
// CHECK-NEXT:    [[LOCALID_:%.*]] = memref.alloca() : memref<1x!sycl_id_3_1>
// CHECK:         memref.store [[LOCALID]], [[LOCALID_:%.*]][%c0{{.*}}] : memref<1x!sycl_id_3_1>
// CHECK:         [[LOCALID0_:%.*]] = sycl.id.get [[LOCALID_]][%c0{{.*}}] : (memref<1x!sycl_id_3_1>, i32) -> memref<?xindex>
// CHECK-NEXT:    [[LOCALID0:%.*]] = memref.load [[LOCALID0_]][%c0{{.*}}] : memref<?xindex>
// CHECK:         [[LOCALID1_:%.*]] = sycl.id.get [[LOCALID_]][%c1{{.*}}] : (memref<1x!sycl_id_3_1>, i32) -> memref<?xindex>
// CHECK-NEXT:    [[LOCALID1:%.*]] = memref.load [[LOCALID1_]][%c0{{.*}}] : memref<?xindex>
// CHECK:         [[LOCALID2_:%.*]] = sycl.id.get [[LOCALID_]][%c2{{.*}}] : (memref<1x!sycl_id_3_1>, i32) -> memref<?xindex>
// CHECK-NEXT:    [[LOCALID2:%.*]] = memref.load [[LOCALID2_]][%c0{{.*}}] : memref<?xindex>

// CHECK:         [[TX:%.*]] = sycl.nd_item.get_global_id(%arg1, %c0{{.*}}) : (memref<?x!sycl_nd_item_3_>, i32) -> i64  
// CHECK-NEXT:    affine.for [[IV1:%.*]] = 0 to 256 {

// COM: Get pointer to local memory:
// CHECK-NEXT:      [[GETGLOBAL:%.*]] = memref.get_global @WGLocalMem : memref<32000xi8, #sycl.access.address_space<local>>

// SIZE1-NEXT:      [[TILESIZE:%.*]] = arith.constant 1 : index
// SIZE2-NEXT:      [[TILESIZE:%.*]] = arith.constant 2 : index
// CHECK-NEXT:      affine.for [[IV2:%.*]] = 1 to [[MAP1]]()[[[TILESIZE]]] {

// COM: Get pointer to memref for 1st load:
// CHECK:             [[ID0:%.*]] = memref.alloca() : memref<1x!sycl_id_3_>
// CHECK:             [[ID0GET0:%.*]] = sycl.id.get [[ID0]][%c0_i32{{.*}}] : (memref<1x!sycl_id_3_>, i32) -> memref<?xindex>
// CHECK:             memref.store [[LOCALID0]], [[ID0GET0]][%c0{{.*}}] : memref<?xindex>
// CHECK:             [[ID0GET1:%.*]] = sycl.id.get [[ID0]][%c1_i32{{.*}}] : (memref<1x!sycl_id_3_>, i32) -> memref<?xindex>
// CHECK:             memref.store [[LOCALID1]], [[ID0GET1]][%c0{{.*}}] : memref<?xindex>
// CHECK:             [[ID0GET2:%.*]] = sycl.id.get [[ID0]][%c2_i32{{.*}}] : (memref<1x!sycl_id_3_>, i32) -> memref<?xindex>
// CHECK:             memref.store [[LOCALID2]], [[ID0GET2]][%c0{{.*}}] : memref<?xindex>
// CHECK:             [[ACCSUB0:%.*]] = sycl.accessor.subscript %arg0[[[ID0]]] : (memref<?x!sycl_accessor_3_f32_r_gb>, memref<1x!sycl_id_3_>) -> memref<?xf32, 1>

// COM: Get pointer to local memory for 1st load:
// CHECK:             [[VIEW0:%.*]] = memref.view [[GETGLOBAL]][%c0{{.*}}][[[LOCALRANGE2_]], [[LOCALRANGE1_]], [[LOCALRANGE0_]]] : memref<32000xi8, #sycl.access.address_space<local>> to memref<?x?x?xf32, #sycl.access.address_space<local>>

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
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
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

// CHECK:       memref.global "private" @WGLocalMem : memref<32000xi8, #sycl.access.address_space<local>>
// CHECK-LABEL: func.func private @scf_2d(%arg0: memref<?x!sycl_accessor_2_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_2_>) {

// COM: Get work group sizes:
// CHECK:         [[LOCALRANGE0:%.*]] = sycl.nd_item.get_local_range(%arg1, %c0{{.*}}) : (memref<?x!sycl_nd_item_2_>, i32) -> i64 
// CHECK-NEXT:    [[LOCALRANGE0_:%.*]] = arith.index_cast [[LOCALRANGE0]] : i64 to index 
// CHECK:         [[LOCALRANGE1:%.*]] = sycl.nd_item.get_local_range(%arg1, %c1{{.*}}) : (memref<?x!sycl_nd_item_2_>, i32) -> i64 
// CHECK-NEXT:    [[LOCALRANGE1_:%.*]] = arith.index_cast [[LOCALRANGE1]] : i64 to index

// COM: Get local ids:
// CHECK:         [[LOCALID:%.*]] = sycl.local_id : !sycl_id_2_1
// CHECK-NEXT:    [[LOCALID_:%.*]] = memref.alloca() : memref<1x!sycl_id_2_1>
// CHECK:         memref.store [[LOCALID]], [[LOCALID_:%.*]][%c0{{.*}}] : memref<1x!sycl_id_2_1>
// CHECK:         [[LOCALID0_:%.*]] = sycl.id.get [[LOCALID_]][%c0{{.*}}] : (memref<1x!sycl_id_2_1>, i32) -> memref<?xindex>
// CHECK-NEXT:    [[LOCALID0:%.*]] = memref.load [[LOCALID0_]][%c0{{.*}}] : memref<?xindex>
// CHECK:         [[LOCALID1_:%.*]] = sycl.id.get [[LOCALID_]][%c1{{.*}}] : (memref<1x!sycl_id_2_1>, i32) -> memref<?xindex>
// CHECK-NEXT:    [[LOCALID1:%.*]] = memref.load [[LOCALID1_]][%c0{{.*}}] : memref<?xindex>

// CHECK:         [[TX:%.*]] = sycl.nd_item.get_global_id(%arg1, %c0_i32{{.*}}) : (memref<?x!sycl_nd_item_2_>, i32) -> i64  

// COM: Get pointer to local memory:
// CHECK-NEXT:    [[GETGLOBAL:%.*]] = memref.get_global @WGLocalMem : memref<32000xi8, #sycl.access.address_space<local>>

// SIZE1-NEXT:    [[TILESIZE:%.*]] = arith.constant 1 : index
// SIZE2-NEXT:    [[TILESIZE:%.*]] = arith.constant 2 : index
// CHECK-NEXT:    [[STEP:%.*]] = arith.muli %c1, [[TILESIZE]] : index
// CHECK-NEXT:    scf.for [[IV1:%.*]] = %c0{{.*}} to %c256 step [[STEP]] {

// COM: Get pointer to memref for 1st load:
// CHECK:           [[ID0:%.*]] = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK:           [[ID0GET0:%.*]] = sycl.id.get [[ID0]][%c0_i32{{.*}}] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
// CHECK:           memref.store [[LOCALID0]], [[ID0GET0]][%c0{{.*}}] : memref<?xindex>
// CHECK:           [[ID0GET1:%.*]] = sycl.id.get [[ID0]][%c1_i32{{.*}}] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
// CHECK:           memref.store [[LOCALID1]], [[ID0GET1]][%c0{{.*}}] : memref<?xindex>
// CHECK:           [[ACCSUB0:%.*]] = sycl.accessor.subscript %arg0[[[ID0]]] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<1x!sycl_id_2_>) -> memref<?xf32, 1>

// COM: Get pointer to local memory for 1st load:
// CHECK:           [[VIEW0:%.*]] = memref.view [[GETGLOBAL]][%c0{{.*}}][[[LOCALRANGE1_]], [[LOCALRANGE0_]]] : memref<32000xi8, #sycl.access.address_space<local>> to memref<?x?xf32, #sycl.access.address_space<local>>

// COM: Compute 2nd load offset:
// CHECK:           [[MUL1:%.*]] = arith.muli %c4, [[LOCALRANGE0_]] : index 
// CHECK:           [[MUL2:%.*]] = arith.muli [[MUL1]], [[LOCALRANGE1_]] : index 
// CHECK:           [[OFFSET:%.*]] = arith.addi %c0{{.*}}, [[MUL2]] : index 

// COM: Get pointer to memref for 2nd load:
// CHECK:           [[ID1:%.*]] = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK:           [[ID1GET0:%.*]] = sycl.id.get [[ID1]][%c0_i32{{.*}}] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
// CHECK:           memref.store [[LOCALID0]], [[ID1GET0]][%c0{{.*}}] : memref<?xindex>
// CHECK:           [[ID1GET1:%.*]] = sycl.id.get [[ID1]][%c1_i32{{.*}}] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
// CHECK:           memref.store [[LOCALID1]], [[ID1GET1]][%c0{{.*}}] : memref<?xindex>
// CHECK:           [[ACCSUB1:%.*]] = sycl.accessor.subscript %arg0[[[ID1]]] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<1x!sycl_id_2_>) -> memref<?xf32, 1>

// COM: Get pointer to local memory for 2nd load:
// CHECK:           [[VIEW1:%.*]] = memref.view [[GETGLOBAL]][[[OFFSET]]][[[LOCALRANGE1_]], [[LOCALRANGE0_]]] : memref<32000xi8, #sycl.access.address_space<local>> to memref<?x?xf32, #sycl.access.address_space<local>>

// CHECK-NEXT:      spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:      [[VAL_1:%.*]] = arith.addi [[IV1]], [[STEP]] : index
// CHECK-NEXT:      [[VAL_2:%.*]] = arith.cmpi slt, %c256, [[VAL_1]] : index
// CHECK-NEXT:      [[VAL_3:%.*]] = arith.select [[VAL_2]], %c256, [[VAL_1]] : index
// CHECK-NEXT:      scf.for [[IV2:%.*]] = [[IV1]] to [[VAL_3]] step %c1 {
// CHECK-NEXT:        [[IV2_CAST:%.*]] = arith.index_cast [[IV2]] : index to i64   
// CHECK-NEXT:        sycl.constructor @id([[ID:%.*]], [[TX]], [[IV2_CAST]]) {{.*}} : (memref<?x!sycl_id_2_>, i64, i64)
// CHECK-NEXT:        [[SUBSCR1:%.*]] = sycl.accessor.subscript %arg0[[[ID]]] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<?x!sycl_id_2_>) -> memref<?xf32>
// CHECK-NEXT:        {{.*}} = affine.load [[SUBSCR1]][0] : memref<?xf32>
// CHECK-NEXT:        [[SUBSCR2:%.*]] = sycl.accessor.subscript %arg0[[[ID]]] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<?x!sycl_id_2_>) -> memref<?xf32>
// CHECK-NEXT:        {{.*}} = affine.load [[SUBSCR2]][0] : memref<?xf32>
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
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %c0_i32 = arith.constant 0 : i32  
  %c1_i32 = arith.constant 1 : i32  
  %tx = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_2>, i32) -> i64

  scf.for %ii = %c0 to %c256 step %c1 {
    %i = arith.index_cast %ii : index to i64    
    sycl.constructor @id(%id, %tx, %i) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2>, i64, i64)
    %subscr1 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<?x!sycl_id_2>) -> memref<?xf32>
    %load1 = affine.load %subscr1[0] : memref<?xf32>
    %subscr2 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<?x!sycl_id_2>) -> memref<?xf32>
    %load2 = affine.load %subscr2[0] : memref<?xf32>
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

// CHECK:       memref.global "private" @WGLocalMem : memref<32000xi8, #sycl.access.address_space<local>>
// CHECK-LABEL: func.func private @scf_3d(%arg0: memref<?x!sycl_accessor_3_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_3_>) {

// COM: Get work group sizes:
// CHECK:         [[LOCALRANGE0:%.*]] = sycl.nd_item.get_local_range(%arg1, %c0{{.*}}) : (memref<?x!sycl_nd_item_3_>, i32) -> i64 
// CHECK-NEXT:    [[LOCALRANGE0_:%.*]] = arith.index_cast [[LOCALRANGE0]] : i64 to index 
// CHECK:         [[LOCALRANGE1:%.*]] = sycl.nd_item.get_local_range(%arg1, %c1{{.*}}) : (memref<?x!sycl_nd_item_3_>, i32) -> i64 
// CHECK-NEXT:    [[LOCALRANGE1_:%.*]] = arith.index_cast [[LOCALRANGE1]] : i64 to index
// CHECK:         [[LOCALRANGE2:%.*]] = sycl.nd_item.get_local_range(%arg1, %c2{{.*}}) : (memref<?x!sycl_nd_item_3_>, i32) -> i64 
// CHECK-NEXT:    [[LOCALRANGE2_:%.*]] = arith.index_cast [[LOCALRANGE2]] : i64 to index

// COM: Get local ids:
// CHECK:         [[LOCALID:%.*]] = sycl.local_id : !sycl_id_3_1
// CHECK-NEXT:    [[LOCALID_:%.*]] = memref.alloca() : memref<1x!sycl_id_3_1>
// CHECK:         memref.store [[LOCALID]], [[LOCALID_:%.*]][%c0{{.*}}] : memref<1x!sycl_id_3_1>
// CHECK:         [[LOCALID0_:%.*]] = sycl.id.get [[LOCALID_]][%c0{{.*}}] : (memref<1x!sycl_id_3_1>, i32) -> memref<?xindex>
// CHECK-NEXT:    [[LOCALID0:%.*]] = memref.load [[LOCALID0_]][%c0{{.*}}] : memref<?xindex>
// CHECK:         [[LOCALID1_:%.*]] = sycl.id.get [[LOCALID_]][%c1{{.*}}] : (memref<1x!sycl_id_3_1>, i32) -> memref<?xindex>
// CHECK-NEXT:    [[LOCALID1:%.*]] = memref.load [[LOCALID1_]][%c0{{.*}}] : memref<?xindex>
// CHECK:         [[LOCALID2_:%.*]] = sycl.id.get [[LOCALID_]][%c2{{.*}}] : (memref<1x!sycl_id_3_1>, i32) -> memref<?xindex>
// CHECK-NEXT:    [[LOCALID2:%.*]] = memref.load [[LOCALID2_]][%c0{{.*}}] : memref<?xindex>

// CHECK:         [[TX:%.*]] = sycl.nd_item.get_global_id(%arg1, %c0{{.*}}) : (memref<?x!sycl_nd_item_3_>, i32) -> i64  
// CHECK-NEXT:    scf.for [[IV1:.*]] = %c0{{.*}} to %c256 step %c1 {

// COM: Get pointer to local memory:
// CHECK-NEXT:      [[GETGLOBAL:%.*]] = memref.get_global @WGLocalMem : memref<32000xi8, #sycl.access.address_space<local>>

// SIZE1-NEXT:      [[TILESIZE:%.*]] = arith.constant 1 : index
// SIZE2-NEXT:      [[TILESIZE:%.*]] = arith.constant 2 : index
// CHECK-NEXT:      [[STEP:%.*]] = arith.muli %c1, [[TILESIZE]] : index
// CHECK-NEXT:      scf.for [[IV2:.*]] = %c1 to %c512 step [[STEP]] {

// COM: Get pointer to memref for 1st load:
// CHECK:             [[ID0:%.*]] = memref.alloca() : memref<1x!sycl_id_3_>
// CHECK:             [[ID0GET0:%.*]] = sycl.id.get [[ID0]][%c0_i32{{.*}}] : (memref<1x!sycl_id_3_>, i32) -> memref<?xindex>
// CHECK:             memref.store [[LOCALID0]], [[ID0GET0]][%c0{{.*}}] : memref<?xindex>
// CHECK:             [[ID0GET1:%.*]] = sycl.id.get [[ID0]][%c1_i32{{.*}}] : (memref<1x!sycl_id_3_>, i32) -> memref<?xindex>
// CHECK:             memref.store [[LOCALID1]], [[ID0GET1]][%c0{{.*}}] : memref<?xindex>
// CHECK:             [[ID0GET2:%.*]] = sycl.id.get [[ID0]][%c2_i32{{.*}}] : (memref<1x!sycl_id_3_>, i32) -> memref<?xindex>
// CHECK:             memref.store [[LOCALID2]], [[ID0GET2]][%c0{{.*}}] : memref<?xindex>
// CHECK:             [[ACCSUB:%.*]] = sycl.accessor.subscript %arg0[[[ID0]]] : (memref<?x!sycl_accessor_3_f32_r_gb>, memref<1x!sycl_id_3_>) -> memref<?xf32, 1>

// COM: Get pointer to local memory for 1st load:
// CHECK:             [[VIEW0:%.*]] = memref.view [[GETGLOBAL]][%c0{{.*}}][[[LOCALRANGE2_]], [[LOCALRANGE1_]], [[LOCALRANGE0_]]] : memref<32000xi8, #sycl.access.address_space<local>> to memref<?x?x?xf32, #sycl.access.address_space<local>>

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
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %c0_i32 = arith.constant 0 : i32  
  %c1_i32 = arith.constant 1 : i32  
  %c2_i32 = arith.constant 2 : i32  
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
