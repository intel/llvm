// RUN: polygeist-opt --loop-internalization="unroll-factor=1" --split-input-file %s | FileCheck %s

!sycl_array_2 = !sycl.array<[2], (memref<2xi64, 4>)>
!sycl_id_2 = !sycl.id<[2], (!sycl_array_2)>
!sycl_range_2 = !sycl.range<[2], (!sycl_array_2)>
!sycl_accessor_impl_device_2 = !sycl.accessor_impl_device<[2], (!sycl_id_2, !sycl_range_2, !sycl_range_2)>
!sycl_group_2 = !sycl.group<[2], (!sycl_range_2, !sycl_range_2, !sycl_range_2, !sycl_id_2)>
!sycl_item_base_2 = !sycl.item_base<[2, true], (!sycl_range_2, !sycl_id_2, !sycl_id_2)>
!sycl_accessor_2_f32_r_dev = !sycl.accessor<[2, f32, read, device], (!sycl_accessor_impl_device_2, !llvm.struct<(memref<?xf32, 2>)>)>
!sycl_item_2 = !sycl.item<[2, true], (!sycl_item_base_2)>
!sycl_nd_item_2 = !sycl.nd_item<[2], (!sycl_item_2, !sycl_item_2, !sycl_group_2)>

// CHECK-DAG:   [[MAP1:#map.*]] = affine_map<()[s0] -> (256 ceildiv s0)>
// CHECK-DAG:   [[MAP2:#map.*]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG:   [[MAP3:#map.*]] = affine_map<(d0)[s0] -> (d0 * s0 + s0, 256)>
// CHECK:       memref.global "private" @WGLocalMem : memref<64xi8, #sycl.access.address_space<local>>
// CHECK-LABEL: func.func private @affine_2d(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<?x!sycl_accessor_2_f32_r_dev>, %[[VAL_1:.*]]: memref<?x!sycl_item_2_>) {

// COM: Get local ids:
// CHECK-NEXT:    %[[VAL_2:.*]] = sycl.local_id : !sycl_id_2_1
// CHECK-NEXT:    %[[VAL_3:.*]] = memref.alloca() : memref<1x!sycl_id_2_1>
// CHECK-NEXT:    %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:    memref.store %[[VAL_2]], %[[VAL_3]]{{\[}}%[[VAL_4]]] : memref<1x!sycl_id_2_1>
// CHECK-NEXT:    %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[LOCALID0:.*]] = sycl.id.get %[[VAL_3]]{{\[}}%[[VAL_5]]] : (memref<1x!sycl_id_2_1>, i32) -> index
// CHECK-NEXT:    %[[VAL_8:.*]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[LOCALID1:.*]] = sycl.id.get %[[VAL_3]]{{\[}}%[[VAL_8]]] : (memref<1x!sycl_id_2_1>, i32) -> index

// COM: Original code:
// CHECK-NEXT:    %[[VAL_11:.*]] = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT:    %[[VAL_12:.*]] = memref.cast %[[VAL_11]] : memref<1x!sycl_id_2_> to memref<?x!sycl_id_2_>
// CHECK-NEXT:    %[[VAL_13:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[VAL_14:.*]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[VAL_15:.*]] = sycl.item.get_id(%[[VAL_1]], %[[VAL_13]]) : (memref<?x!sycl_item_2_>, i32) -> i64
// CHECK-NEXT:    %[[VAL_16:.*]] = arith.index_cast %[[VAL_15]] : i64 to index

// COM: Get pointer to shared local memory:
// CHECK-NEXT:    %[[VAL_17:.*]] = memref.get_global @WGLocalMem : memref<64xi8, #sycl.access.address_space<local>>

// COM: Use work group size of dimension 1 as tile size:
// CHECK-NEXT:    %[[TILESIZE:.*]] = arith.constant 4 : index
// CHECK-NEXT:    affine.for %[[VAL_19:.*]] = 0 to [[MAP1]](){{\[}}%[[TILESIZE]]] {

// COM: Get pointer to the shared local memory portion for 1st memref:
// CHECK-NEXT:      %[[VAL_20:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_21:.*]] = memref.view %[[VAL_17]]{{\[}}%[[VAL_20]]][] : memref<64xi8, #sycl.access.address_space<local>> to memref<2x4xf32, #sycl.access.address_space<local>>

// COM: Get tiled loop lower bound:
// CHECK-NEXT:      %[[VAL_22:.*]] = affine.apply [[MAP2]](%[[VAL_19]]){{\[}}%[[TILESIZE]]]

// COM: Calculate indexes for global memory:
// CHECK-NEXT:      %[[VAL_23:.*]] = arith.addi %[[LOCALID1]], %[[VAL_22]] : index
// CHECK-NEXT:      %[[VAL_24:.*]] = arith.index_cast %[[VAL_23]] : index to i64
// CHECK-NEXT:      %[[VAL_25:.*]] = arith.addi %[[LOCALID1]], %[[VAL_22]] : index
// CHECK-NEXT:      %[[VAL_26:.*]] = arith.index_cast %[[VAL_25]] : index to i64

// COM: Copy to shared local memory for 1st memref:
// CHECK-NEXT:      %[[VAL_27:.*]] = sycl.id.constructor(%[[VAL_16]], %[[VAL_25]]) : (index, index) -> memref<1x!sycl_id_2_>
// CHECK-NEXT:      %[[VAL_28:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_29:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_27]]] : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<1x!sycl_id_2_>) -> memref<?xf32, 1>
// CHECK-NEXT:      %[[VAL_30:.*]] = memref.load %[[VAL_29]]{{\[}}%[[VAL_28]]] : memref<?xf32, 1>
// CHECK-NEXT:      memref.store %[[VAL_30]], %[[VAL_21]]{{\[}}%[[LOCALID0]], %[[LOCALID1]]] : memref<2x4xf32, #sycl.access.address_space<local>>

// COM: Get pointer to the shared local memory portion for 2nd memref:
// CHECK-NEXT:      %[[VAL_36:.*]] = arith.constant 32 : index
// CHECK-NEXT:      %[[VAL_37:.*]] = memref.view %[[VAL_17]]{{\[}}%[[VAL_36]]][] : memref<64xi8, #sycl.access.address_space<local>> to memref<2x4xf32, #sycl.access.address_space<local>>

// COM: Copy to shared local memory for 2nd memref:
// CHECK-NEXT:      %[[VAL_31:.*]] = sycl.id.constructor(%[[VAL_16]], %[[VAL_23]]) : (index, index) -> memref<1x!sycl_id_2_>
// CHECK-NEXT:      %[[VAL_32:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_33:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_31]]] : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<1x!sycl_id_2_>) -> memref<?xf32, 1>
// CHECK-NEXT:      %[[VAL_34:.*]] = memref.load %[[VAL_33]]{{\[}}%[[VAL_32]]] : memref<?xf32, 1>
// CHECK-NEXT:      memref.store %[[VAL_34]], %[[VAL_37]]{{\[}}%[[LOCALID0]], %[[LOCALID1]]] : memref<2x4xf32, #sycl.access.address_space<local>>

// CHECK-NEXT:      spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:      affine.for %[[VAL_47:.*]] = [[MAP2]](%[[VAL_19]]){{\[}}%[[TILESIZE]]] to min [[MAP3]](%[[VAL_19]]){{\[}}%[[TILESIZE]]] {
// CHECK-NEXT:        %[[VAL_48:.*]] = arith.subi %[[VAL_47]], %[[VAL_22]] : index
// CHECK-NEXT:        %[[VAL_49:.*]] = arith.index_cast %[[VAL_47]] : index to i64
// CHECK-NEXT:        %[[VAL_50:.*]] = arith.index_cast %[[VAL_48]] : index to i64
// CHECK-NEXT:        %[[VAL_51:.*]] = arith.index_cast %[[VAL_48]] : index to i64
// COM: TODO: sycl.constructor can be removed.
// CHECK-NEXT:        sycl.constructor @id(%[[VAL_12]], %[[VAL_15]], %[[VAL_49]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2_>, i64, i64)
// CHECK-DAG:         %[[VAL_52:.*]] = memref.load %[[VAL_21]]{{\[}}%[[LOCALID0]], %[[VAL_48]]] : memref<2x4xf32, #sycl.access.address_space<local>>
// CHECK-DAG:         %[[VAL_53:.*]] = memref.load %[[VAL_37]]{{\[}}%[[LOCALID0]], %[[VAL_48]]] : memref<2x4xf32, #sycl.access.address_space<local>>
// CHECK-NEXT:      }
// CHECK-NEXT:      spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
gpu.module @device_func {
func.func private @affine_2d(%arg0: memref<?x!sycl_accessor_2_f32_r_dev>, %arg1: memref<?x!sycl_item_2>) {
  %alloca = memref.alloca() : memref<1x!sycl_id_2>
  %id = memref.cast %alloca : memref<1x!sycl_id_2> to memref<?x!sycl_id_2>
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %tx = sycl.item.get_id(%arg1, %c0_i32) : (memref<?x!sycl_item_2>, i32) -> i64

  affine.for %ii = 0 to 256 {
    %i = arith.index_cast %ii : index to i64
    sycl.constructor @id(%id, %tx, %i) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2>, i64, i64)
    %subscr1 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<?x!sycl_id_2>) -> memref<?xf32>
    %load1 = affine.load %subscr1[0] : memref<?xf32>
    %subscr2 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<?x!sycl_id_2>) -> memref<?xf32>
    %load2 = affine.load %subscr2[0] : memref<?xf32>
  }
  return
}
gpu.func @kernel(%arg0: memref<?x!sycl_accessor_2_f32_r_dev>, %arg1: memref<?x!sycl_item_2>) kernel attributes {reqd_work_group_size = [4, 2]} {
  func.call @affine_2d(%arg0, %arg1) : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<?x!sycl_item_2>) -> ()
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
!sycl_accessor_3_f32_r_dev = !sycl.accessor<[3, f32, read, device], (!sycl_accessor_impl_device_3, !llvm.struct<(memref<?xf32, 3>)>)>
!sycl_item_3 = !sycl.item<[3, true], (!sycl_item_base_3)>
!sycl_nd_item_3 = !sycl.nd_item<[3], (!sycl_item_3, !sycl_item_3, !sycl_group_3)>

// CHECK-DAG:   [[MAP1:#map.*]] = affine_map<()[s0] -> (511 ceildiv s0 + 1)>
// CHECK-DAG:   [[MAP2:#map.*]] = affine_map<(d0)[s0] -> ((d0 - 1) * s0 + 1)>
// CHECK-DAG:   [[MAP3:#map.*]] = affine_map<(d0)[s0] -> ((d0 - 1) * s0 + s0 + 1, 512)>
// CHECK:       memref.global "private" @WGLocalMem : memref<32000xi8, #sycl.access.address_space<local>>
// CHECK-LABEL:  func.func private @affine_3d(
// CHECK-SAME:       %[[VAL_0:.*]]: memref<?x!sycl_accessor_3_f32_r_dev>, %[[VAL_1:.*]]: memref<?x!sycl_nd_item_3_>) {

// COM: Get work group sizes:
// CHECK-NEXT:    %[[VAL_2:.*]] = sycl.work_group_size : !sycl_range_3_1
// CHECK-NEXT:    %[[VAL_3:.*]] = memref.alloca() : memref<1x!sycl_range_3_1>
// CHECK-NEXT:    %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:    memref.store %[[VAL_2]], %[[VAL_3]]{{\[}}%[[VAL_4]]] : memref<1x!sycl_range_3_1>
// CHECK-NEXT:    %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[WGSIZE0:.*]] = sycl.range.get %[[VAL_3]]{{\[}}%[[VAL_5]]] : (memref<1x!sycl_range_3_1>, i32) -> index
// CHECK-NEXT:    %[[VAL_8:.*]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[WGSIZE1:.*]] = sycl.range.get %[[VAL_3]]{{\[}}%[[VAL_8]]] : (memref<1x!sycl_range_3_1>, i32) -> index
// CHECK-NEXT:    %[[VAL_11:.*]] = arith.constant 2 : i32
// CHECK-NEXT:    %[[WGSIZE2:.*]] = sycl.range.get %[[VAL_3]]{{\[}}%[[VAL_11]]] : (memref<1x!sycl_range_3_1>, i32) -> index

// COM: Get shared local memory required:
// COM:   for each loop, get the max of:
// COM:     for each accessor subscript, add:
// COM:       multiply element size (4) with work group size for each dimension
// CHECK-NEXT:    %[[VAL_14:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[VAL_15:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[VAL_16:.*]] = arith.constant 4 : index
// CHECK-NEXT:    %[[VAL_17:.*]] = arith.muli %[[VAL_16]], %[[WGSIZE0]] : index
// CHECK-NEXT:    %[[VAL_18:.*]] = arith.muli %[[VAL_17]], %[[WGSIZE1]] : index
// CHECK-NEXT:    %[[VAL_19:.*]] = arith.muli %[[VAL_18]], %[[WGSIZE2]] : index
// CHECK-NEXT:    %[[VAL_20:.*]] = arith.addi %[[VAL_15]], %[[VAL_19]] : index
// CHECK-NEXT:    %[[REQD_SHARED_MEM:.*]] = arith.maxsi %[[VAL_14]], %[[VAL_20]] : index

// COM: Get local ids:
// CHECK-NEXT:    %[[VAL_22:.*]] = sycl.local_id : !sycl_id_3_1
// CHECK-NEXT:    %[[VAL_23:.*]] = memref.alloca() : memref<1x!sycl_id_3_1>
// CHECK-NEXT:    %[[VAL_24:.*]] = arith.constant 0 : index
// CHECK-NEXT:    memref.store %[[VAL_22]], %[[VAL_23]]{{\[}}%[[VAL_24]]] : memref<1x!sycl_id_3_1>
// CHECK-NEXT:    %[[VAL_25:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[LOCALID0:.*]] = sycl.id.get %[[VAL_23]]{{\[}}%[[VAL_25]]] : (memref<1x!sycl_id_3_1>, i32) -> index
// CHECK-NEXT:    %[[VAL_28:.*]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[LOCALID1:.*]] = sycl.id.get %[[VAL_23]]{{\[}}%[[VAL_28]]] : (memref<1x!sycl_id_3_1>, i32) -> index
// CHECK-NEXT:    %[[VAL_31:.*]] = arith.constant 2 : i32
// CHECK-NEXT:    %[[LOCALID2:.*]] = sycl.id.get %[[VAL_23]]{{\[}}%[[VAL_31]]] : (memref<1x!sycl_id_3_1>, i32) -> index

// COM: Original code:
// CHECK-NEXT:    %[[VAL_34:.*]] = memref.alloca() : memref<1x!sycl_id_3_>
// CHECK-NEXT:    %[[VAL_35:.*]] = memref.cast %[[VAL_34]] : memref<1x!sycl_id_3_> to memref<?x!sycl_id_3_>
// CHECK-NEXT:    %[[VAL_36:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[VAL_37:.*]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[VAL_38:.*]] = arith.constant 2 : i32
// CHECK-NEXT:    %[[VAL_39:.*]] = sycl.nd_item.get_global_id(%[[VAL_1]], %[[VAL_36]]) : (memref<?x!sycl_nd_item_3_>, i32) -> i64
// CHECK-NEXT:    %[[VAL_40:.*]] = arith.index_cast %[[VAL_39]] : i64 to index
// CHECK-NEXT:    %[[VAL_41:.*]] = sycl.nd_item.get_global_id(%[[VAL_1]], %[[VAL_37]]) : (memref<?x!sycl_nd_item_3_>, i32) -> i64
// CHECK-NEXT:    %[[VAL_42:.*]] = sycl.nd_item.get_global_id(%[[VAL_1]], %[[VAL_38]]) : (memref<?x!sycl_nd_item_3_>, i32) -> i64
// CHECK-NEXT:    affine.for %[[VAL_43:.*]] = 0 to 256 {

// COM: Ensure there is a sufficient amount of shared local memory available:
// CHECK-NEXT:    %[[SHARED_MEM_AMOUNT:.*]] = arith.constant 32000 : index
// CHECK-NEXT:    %[[VER_COND:.*]] = arith.cmpi ule, %[[REQD_SHARED_MEM]], %[[SHARED_MEM_AMOUNT]] : index
// CHECK-NEXT:    scf.if %[[VER_COND]] {

// COM: Get pointer to shared local memory:
// CHECK-NEXT:      %[[VAL_44:.*]] = memref.get_global @WGLocalMem : memref<32000xi8, #sycl.access.address_space<local>>

// COM: Use work group size of dimension 2 as tile size:
// CHECK-NEXT:      affine.for %[[VAL_45:.*]] = 1 to [[MAP1]](){{\[}}%[[WGSIZE2]]] {

// COM: Get pointer to the shared local memory portion for 1st memref:
// CHECK-NEXT:        %[[VAL_46:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_47:.*]] = memref.view %[[VAL_44]]{{\[}}%[[VAL_46]]]{{\[}}%[[WGSIZE0]], %[[WGSIZE1]], %[[WGSIZE2]]] : memref<32000xi8, #sycl.access.address_space<local>> to memref<?x?x?xf32, #sycl.access.address_space<local>>

// COM: Calculate indexes for global memory:
// CHECK-NEXT:        %[[VAL_48:.*]] = arith.index_cast %[[VAL_43]] : index to i64
// CHECK-NEXT:        %[[VAL_49:.*]] = affine.apply [[MAP2]](%[[VAL_45]]){{\[}}%[[WGSIZE2]]]
// CHECK-NEXT:        %[[VAL_50:.*]] = arith.addi %[[LOCALID2]], %[[VAL_49]] : index
// CHECK-NEXT:        %[[VAL_51:.*]] = arith.index_cast %[[VAL_50]] : index to i64

// COM: Copy to shared local memory for 1st memref:
// CHECK-NEXT:        %[[VAL_52:.*]] = sycl.id.constructor(%[[VAL_40]], %[[VAL_43]], %[[VAL_50]]) : (index, index, index) -> memref<1x!sycl_id_3_>
// CHECK-NEXT:        %[[VAL_53:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_54:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_52]]] : (memref<?x!sycl_accessor_3_f32_r_dev>, memref<1x!sycl_id_3_>) -> memref<?xf32, 1>
// CHECK-NEXT:        %[[VAL_55:.*]] = memref.load %[[VAL_54]]{{\[}}%[[VAL_53]]] : memref<?xf32, 1>
// CHECK-NEXT:        memref.store %[[VAL_55]], %[[VAL_47]]{{\[}}%[[LOCALID0]], %[[LOCALID1]], %[[LOCALID2]]] : memref<?x?x?xf32, #sycl.access.address_space<local>>

// CHECK-NEXT:        spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:        affine.for %[[VAL_63:.*]] = [[MAP2]](%[[VAL_45]]){{\[}}%[[WGSIZE2]]] to min [[MAP3]](%[[VAL_45]]){{\[}}%[[WGSIZE2]]] {
// CHECK-NEXT:          %[[VAL_64:.*]] = arith.subi %[[VAL_63]], %[[VAL_49]] : index
// CHECK-NEXT:          %[[VAL_65:.*]] = arith.index_cast %[[VAL_63]] : index to i64
// CHECK-NEXT:          %[[VAL_66:.*]] = arith.index_cast %[[VAL_64]] : index to i64
// CHECK-NEXT:          sycl.constructor @id(%[[VAL_35]], %[[VAL_39]], %[[VAL_48]], %[[VAL_65]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_3_>, i64, i64, i64)
// CHECK-NEXT:          %[[VAL_67:.*]] = memref.load %[[VAL_47]]{{\[}}%[[LOCALID0]], %[[LOCALID1]], %[[VAL_64]]] : memref<?x?x?xf32, #sycl.access.address_space<local>>
// CHECK-NEXT:          sycl.constructor @id(%[[VAL_35]], %[[VAL_39]], %[[VAL_41]], %[[VAL_42]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_3_>, i64, i64, i64)
// CHECK-NEXT:          %[[VAL_68:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_35]]] : (memref<?x!sycl_accessor_3_f32_r_dev>, memref<?x!sycl_id_3_>) -> memref<?xf32>
// CHECK-NEXT:          %[[VAL_69:.*]] = affine.load %[[VAL_68]][0] : memref<?xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:      }
// CHECK-NEXT:    } else {
// CHECK-NEXT:      affine.for %[[VAL_72:.*]] = 1 to 512 {
// CHECK-NEXT:        %[[VAL_73:.*]] = arith.index_cast %[[VAL_43]] : index to i64
// CHECK-NEXT:        %[[VAL_74:.*]] = arith.index_cast %[[VAL_72]] : index to i64
// CHECK-NEXT:        sycl.constructor @id(%[[VAL_35]], %[[VAL_39]], %[[VAL_73]], %[[VAL_74]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_3_>, i64, i64, i64)
// CHECK-NEXT:        %[[VAL_75:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_35]]] : (memref<?x!sycl_accessor_3_f32_r_dev>, memref<?x!sycl_id_3_>) -> memref<?xf32>
// CHECK-NEXT:        %[[VAL_76:.*]] = affine.load %[[VAL_75]][0] : memref<?xf32>
// CHECK-NEXT:        sycl.constructor @id(%[[VAL_35]], %[[VAL_39]], %[[VAL_41]], %[[VAL_42]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_3_>, i64, i64, i64)
// CHECK-NEXT:        %[[VAL_77:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_35]]] : (memref<?x!sycl_accessor_3_f32_r_dev>, memref<?x!sycl_id_3_>) -> memref<?xf32>
// CHECK-NEXT:        %[[VAL_78:.*]] = affine.load %[[VAL_77]][0] : memref<?xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
gpu.module @device_func {
func.func private @affine_3d(%arg0: memref<?x!sycl_accessor_3_f32_r_dev>, %arg1: memref<?x!sycl_nd_item_3>) {
  %alloca = memref.alloca() : memref<1x!sycl_id_3>
  %id = memref.cast %alloca : memref<1x!sycl_id_3> to memref<?x!sycl_id_3>
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %tx = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_3>, i32) -> i64
  %ty = sycl.nd_item.get_global_id(%arg1, %c1_i32) : (memref<?x!sycl_nd_item_3>, i32) -> i64
  %tz = sycl.nd_item.get_global_id(%arg1, %c2_i32) : (memref<?x!sycl_nd_item_3>, i32) -> i64

  affine.for %ii = 0 to 256 {
    affine.for %jj = 1 to 512 {
      %i = arith.index_cast %ii : index to i64
      %j = arith.index_cast %jj : index to i64

      // Should use shared memory (access exhibits temporal locality).
      sycl.constructor @id(%id, %tx, %i, %j) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_3>, i64, i64, i64)
      %subscr1 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_3_f32_r_dev>, memref<?x!sycl_id_3>) -> memref<?xf32>
      %load1 = affine.load %subscr1[0] : memref<?xf32>

      // Should use global memory (access is coalesced).
      sycl.constructor @id(%id, %tx, %ty, %tz) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_3>, i64, i64, i64)
      %subscr2 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_3_f32_r_dev>, memref<?x!sycl_id_3>) -> memref<?xf32>      
      %load2 = affine.load %subscr2[0] : memref<?xf32>      
    }
  }
  return
}
gpu.func @kernel(%arg0: memref<?x!sycl_accessor_3_f32_r_dev>, %arg1: memref<?x!sycl_nd_item_3>) kernel {
  func.call @affine_3d(%arg0, %arg1) : (memref<?x!sycl_accessor_3_f32_r_dev>, memref<?x!sycl_nd_item_3>) -> ()
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
!sycl_accessor_2_f32_r_dev = !sycl.accessor<[2, f32, read, device], (!sycl_accessor_impl_device_2, !llvm.struct<(memref<?xf32, 2>)>)>
!sycl_item_2 = !sycl.item<[2, true], (!sycl_item_base_2)>
!sycl_nd_item_2 = !sycl.nd_item<[2], (!sycl_item_2, !sycl_item_2, !sycl_group_2)>

// CHECK:           memref.global "private" @WGLocalMem : memref<32000xi8, #sycl.access.address_space<local>>
// CHECK-LABEL:     func.func private @scf_2d(
// CHECK-SAME:          %[[VAL_0:.*]]: memref<?x!sycl_accessor_2_f32_r_dev>, %[[VAL_1:.*]]: memref<?x!sycl_nd_item_2_>) {

// COM: Get work group sizes:
// CHECK-NEXT:        %[[VAL_2:.*]] = sycl.work_group_size : !sycl_range_2_1
// CHECK-NEXT:        %[[VAL_3:.*]] = memref.alloca() : memref<1x!sycl_range_2_1>
// CHECK-NEXT:        %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:        memref.store %[[VAL_2]], %[[VAL_3]]{{\[}}%[[VAL_4]]] : memref<1x!sycl_range_2_1>
// CHECK-NEXT:        %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[WGSIZE0:.*]] = sycl.range.get %[[VAL_3]]{{\[}}%[[VAL_5]]] : (memref<1x!sycl_range_2_1>, i32) -> index
// CHECK-NEXT:        %[[VAL_7:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[WGSIZE1:.*]] = sycl.range.get %[[VAL_3]]{{\[}}%[[VAL_7]]] : (memref<1x!sycl_range_2_1>, i32) -> index

// COM: Get local memory required:
// CHECK-NEXT:        %[[VAL_9:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_11:.*]] = arith.constant 4 : index
// CHECK-NEXT:        %[[VAL_12:.*]] = arith.muli %[[VAL_11]], %[[WGSIZE0]] : index
// CHECK-NEXT:        %[[VAL_13:.*]] = arith.muli %[[VAL_12]], %[[WGSIZE1]] : index
// CHECK-NEXT:        %[[VAL_14:.*]] = arith.addi %[[VAL_10]], %[[VAL_13]] : index
// CHECK-NEXT:        %[[VAL_15:.*]] = arith.constant 4 : index
// CHECK-NEXT:        %[[VAL_16:.*]] = arith.muli %[[VAL_15]], %[[WGSIZE0]] : index
// CHECK-NEXT:        %[[VAL_17:.*]] = arith.muli %[[VAL_16]], %[[WGSIZE1]] : index
// CHECK-NEXT:        %[[VAL_18:.*]] = arith.addi %[[VAL_14]], %[[VAL_17]] : index
// CHECK-NEXT:        %[[REQD_SHARED_MEM:.*]] = arith.maxsi %[[VAL_9]], %[[VAL_18]] : index

// COM: Get local ids:
// CHECK-NEXT:        %[[VAL_20:.*]] = sycl.local_id : !sycl_id_2_1
// CHECK-NEXT:        %[[VAL_21:.*]] = memref.alloca() : memref<1x!sycl_id_2_1>
// CHECK-NEXT:        %[[VAL_22:.*]] = arith.constant 0 : index
// CHECK-NEXT:        memref.store %[[VAL_20]], %[[VAL_21]]{{\[}}%[[VAL_22]]] : memref<1x!sycl_id_2_1>
// CHECK-NEXT:        %[[VAL_23:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[LOCALID0:.*]] = sycl.id.get %[[VAL_21]]{{\[}}%[[VAL_23]]] : (memref<1x!sycl_id_2_1>, i32) -> index
// CHECK-NEXT:        %[[VAL_25:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[LOCALID1:.*]] = sycl.id.get %[[VAL_21]]{{\[}}%[[VAL_25]]] : (memref<1x!sycl_id_2_1>, i32) -> index

// COM: Original code:
// CHECK-NEXT:        %[[VAL_27:.*]] = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT:        %[[VAL_28:.*]] = memref.cast %[[VAL_27]] : memref<1x!sycl_id_2_> to memref<?x!sycl_id_2_>
// CHECK-NEXT:        %[[VAL_29:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_30:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_31:.*]] = arith.constant 256 : index
// CHECK-NEXT:        %[[VAL_32:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[VAL_33:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_34:.*]] = sycl.nd_item.get_global_id(%[[VAL_1]], %[[VAL_32]]) : (memref<?x!sycl_nd_item_2_>, i32) -> i64
// CHECK-NEXT:        %[[VAL_35:.*]] = arith.index_cast %[[VAL_34]] : i64 to index
// CHECK-NEXT:        %[[VAL_36:.*]] = sycl.nd_item.get_global_id(%[[VAL_1]], %[[VAL_33]]) : (memref<?x!sycl_nd_item_2_>, i32) -> i64
// CHECK-NEXT:        %[[GLOBALID1:.*]] = arith.index_cast %[[VAL_36]] : i64 to index

// COM: Ensure there is a sufficient amount of shared local memory available and memory accesses reference the loop IV 'consistently':
// CHECK-NEXT:        %[[SHARED_MEM_AMOUNT:.*]] = arith.constant 32000 : index
// CHECK-NEXT:        %[[VAL_39:.*]] = arith.cmpi ule, %[[REQD_SHARED_MEM]], %[[SHARED_MEM_AMOUNT]] : index
// CHECK-NEXT:        %[[VAL_40:.*]] = arith.cmpi eq, %[[WGSIZE0]], %[[WGSIZE1]] : index
// CHECK-NEXT:        %[[VER_COND:.*]] = arith.andi %[[VAL_39]], %[[VAL_40]] : i1
// CHECK-NEXT:        scf.if %[[VER_COND]] {

// COM: Get pointer to shared local memory:
// CHECK-NEXT:          %[[VAL_42:.*]] = memref.get_global @WGLocalMem : memref<32000xi8, #sycl.access.address_space<local>>

// COM: Use work group size as tile size:
// CHECK-NEXT:          %[[TILESIZE:.*]] = arith.muli %[[VAL_30]], %[[WGSIZE0]] : index
// CHECK-NEXT:          scf.for %[[VAL_44:.*]] = %[[VAL_29]] to %[[VAL_31]] step %[[TILESIZE]] {

// COM: Calculate indexes for global memory:
// CHECK-NEXT:            %[[VAL_45:.*]] = arith.addi %[[LOCALID0]], %[[VAL_44]] : index
// CHECK-NEXT:            %[[VAL_46:.*]] = arith.index_cast %[[VAL_45]] : index to i64
// CHECK-NEXT:            %[[VAL_47:.*]] = arith.addi %[[LOCALID1]], %[[VAL_44]] : index
// CHECK-NEXT:            %[[VAL_48:.*]] = arith.index_cast %[[VAL_47]] : index to i64
// CHECK-NEXT:            %[[VAL_49:.*]] = arith.constant 0 : index

// COM: Get pointer to the shared local memory portion for 1st memref:
// CHECK-NEXT:            %[[VAL_50:.*]] = memref.view %[[VAL_42]]{{\[}}%[[VAL_49]]]{{\[}}%[[WGSIZE0]], %[[WGSIZE1]]] : memref<32000xi8, #sycl.access.address_space<local>> to memref<?x?xf32, #sycl.access.address_space<local>>

// COM: Calculate upper bound for the tiled loop:
// CHECK-NEXT:            %[[VAL_51:.*]] = arith.addi %[[VAL_44]], %[[TILESIZE]] : index
// CHECK-NEXT:            %[[VAL_52:.*]] = arith.cmpi slt, %[[VAL_31]], %[[VAL_51]] : index
// CHECK-NEXT:            %[[VAL_53:.*]] = arith.select %[[VAL_52]], %[[VAL_31]], %[[VAL_51]] : index

// COM: Copy to shared local memory for 1st memref:
// CHECK-NEXT:            %[[VAL_54:.*]] = sycl.id.constructor(%[[VAL_35]], %[[VAL_47]]) : (index, index) -> memref<1x!sycl_id_2_>
// CHECK-NEXT:            %[[VAL_55:.*]] = arith.constant 0 : index
// CHECK-NEXT:            %[[VAL_56:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_54]]] : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<1x!sycl_id_2_>) -> memref<?xf32, 1>
// CHECK-NEXT:            %[[VAL_57:.*]] = memref.load %[[VAL_56]]{{\[}}%[[VAL_55]]] : memref<?xf32, 1>
// CHECK-NEXT:            memref.store %[[VAL_57]], %[[VAL_50]]{{\[}}%[[LOCALID0]], %[[LOCALID1]]] : memref<?x?xf32, #sycl.access.address_space<local>>

// COM: Calculate offset:
// CHECK-NEXT:            %[[VAL_63:.*]] = arith.constant 4 : index
// CHECK-NEXT:            %[[VAL_64:.*]] = arith.muli %[[VAL_63]], %[[WGSIZE0]] : index
// CHECK-NEXT:            %[[VAL_65:.*]] = arith.muli %[[VAL_64]], %[[WGSIZE1]] : index
// CHECK-NEXT:            %[[VAL_66:.*]] = arith.addi %[[VAL_49]], %[[VAL_65]] : index

// COM: Get pointer to the shared local memory portion for 2nd memref:
// CHECK-NEXT:            %[[VAL_67:.*]] = memref.view %[[VAL_42]]{{\[}}%[[VAL_66]]]{{\[}}%[[WGSIZE1]], %[[WGSIZE0]]] : memref<32000xi8, #sycl.access.address_space<local>> to memref<?x?xf32, #sycl.access.address_space<local>>

// COM: Copy to local memory for 2nd memref:
// CHECK:                 %[[VAL_68:.*]] = sycl.id.constructor(%[[GLOBALID1]], %[[VAL_45]]) : (index, index) -> memref<1x!sycl_id_2_>
// CHECK:                 %[[VAL_69:.*]] = arith.constant 0 : index
// CHECK:                 %[[VAL_70:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_68]]] : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<1x!sycl_id_2_>) -> memref<?xf32, 1>
// CHECK:                 %[[VAL_71:.*]] = memref.load %[[VAL_70]]{{\[}}%[[VAL_69]]] : memref<?xf32, 1>
// CHECK:                 memref.store %[[VAL_71]], %[[VAL_67]]{{\[}}%[[LOCALID1]], %[[LOCALID0]]] : memref<?x?xf32, #sycl.access.address_space<local>>

// CHECK-NEXT:            spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:            scf.for %[[VAL_77:.*]] = %[[VAL_44]] to %[[VAL_53]] step %[[VAL_30]] {
// CHECK-NEXT:              %[[VAL_78:.*]] = arith.subi %[[VAL_77]], %[[VAL_44]] : index
// CHECK-NEXT:              %[[VAL_79:.*]] = arith.index_cast %[[VAL_77]] : index to i64
// CHECK-NEXT:              %[[VAL_80:.*]] = arith.index_cast %[[VAL_78]] : index to i64
// CHECK-NEXT:              %[[VAL_81:.*]] = arith.index_cast %[[VAL_78]] : index to i64
// CHECK-NEXT:              sycl.constructor @id(%[[VAL_28]], %[[VAL_34]], %[[VAL_79]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2_>, i64, i64)
// CHECK-NEXT:              %[[VAL_82:.*]] = memref.load %[[VAL_50]]{{\[}}%[[LOCALID0]], %[[VAL_78]]] : memref<?x?xf32, #sycl.access.address_space<local>>
// CHECK-NEXT:              sycl.constructor @id(%[[VAL_28]], %[[VAL_36]], %[[VAL_79]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2_>, i64, i64)
// CHECK-NEXT:              %[[VAL_83:.*]] = memref.load %[[VAL_67]]{{\[}}%[[LOCALID1]], %[[VAL_78]]] : memref<?x?xf32, #sycl.access.address_space<local>>
// CHECK-NEXT:            }
// CHECK-NEXT:            spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:          }
// CHECK-NEXT:        } else {
// CHECK-NEXT:          scf.for %[[VAL_84:.*]] = %[[VAL_29]] to %[[VAL_31]] step %[[VAL_30]] {
// CHECK-NEXT:            %[[VAL_85:.*]] = arith.index_cast %[[VAL_84]] : index to i64
// CHECK-NEXT:            sycl.constructor @id(%[[VAL_28]], %[[VAL_34]], %[[VAL_85]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2_>, i64, i64)
// CHECK-NEXT:            %[[VAL_86:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_28]]] : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<?x!sycl_id_2_>) -> memref<?xf32>
// CHECK-NEXT:            %[[VAL_87:.*]] = affine.load %[[VAL_86]][0] : memref<?xf32>
// CHECK-NEXT:            sycl.constructor @id(%[[VAL_28]], %[[VAL_36]], %[[VAL_85]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2_>, i64, i64)
// CHECK-NEXT:            %[[VAL_88:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_28]]] : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<?x!sycl_id_2_>) -> memref<?xf32>
// CHECK-NEXT:            %[[VAL_89:.*]] = affine.load %[[VAL_88]][0] : memref<?xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        return
// CHECK-NEXT:      }
gpu.module @device_func {
func.func private @scf_2d(%arg0: memref<?x!sycl_accessor_2_f32_r_dev>, %arg1: memref<?x!sycl_nd_item_2>) {
  %alloca = memref.alloca() : memref<1x!sycl_id_2>
  %id = memref.cast %alloca : memref<1x!sycl_id_2> to memref<?x!sycl_id_2>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %c0_i32 = arith.constant 0 : i32  
  %c1_i32 = arith.constant 1 : i32  
  %tx = sycl.nd_item.get_global_id(%arg1, %c0_i32) : (memref<?x!sycl_nd_item_2>, i32) -> i64
  %ty = sycl.nd_item.get_global_id(%arg1, %c1_i32) : (memref<?x!sycl_nd_item_2>, i32) -> i64

  scf.for %ii = %c0 to %c256 step %c1 {
    %i = arith.index_cast %ii : index to i64    
    sycl.constructor @id(%id, %tx, %i) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2>, i64, i64)
    %subscr1 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<?x!sycl_id_2>) -> memref<?xf32>
    %load1 = affine.load %subscr1[0] : memref<?xf32>

    sycl.constructor @id(%id, %ty, %i) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2>, i64, i64)    
    %subscr2 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<?x!sycl_id_2>) -> memref<?xf32>
    %load2 = affine.load %subscr2[0] : memref<?xf32>
  }
  return
}
gpu.func @kernel(%arg0: memref<?x!sycl_accessor_2_f32_r_dev>, %arg1: memref<?x!sycl_nd_item_2>) kernel {
  func.call @scf_2d(%arg0, %arg1) : (memref<?x!sycl_accessor_2_f32_r_dev>, memref<?x!sycl_nd_item_2>) -> ()
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
!sycl_accessor_3_f32_r_dev = !sycl.accessor<[3, f32, read, device], (!sycl_accessor_impl_device_3, !llvm.struct<(memref<?xf32, 3>)>)>
!sycl_item_3 = !sycl.item<[3, true], (!sycl_item_base_3)>
!sycl_nd_item_3 = !sycl.nd_item<[3], (!sycl_item_3, !sycl_item_3, !sycl_group_3)>

// CHECK:           memref.global "private" @WGLocalMem : memref<32000xi8, #sycl.access.address_space<local>>
// CHECK-LABEL:     func.func private @scf_3d(
// CHECK-SAME:          %[[VAL_0:.*]]: memref<?x!sycl_accessor_3_f32_r_dev>, %[[VAL_1:.*]]: memref<?x!sycl_nd_item_3_>) {

// COM: Get work group sizes:
// CHECK-NEXT:        %[[VAL_2:.*]] = sycl.work_group_size : !sycl_range_3_1
// CHECK-NEXT:        %[[VAL_3:.*]] = memref.alloca() : memref<1x!sycl_range_3_1>
// CHECK-NEXT:        %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:        memref.store %[[VAL_2]], %[[VAL_3]]{{\[}}%[[VAL_4]]] : memref<1x!sycl_range_3_1>
// CHECK-NEXT:        %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[WGSIZE0:.*]] = sycl.range.get %[[VAL_3]]{{\[}}%[[VAL_5]]] : (memref<1x!sycl_range_3_1>, i32) -> index
// CHECK-NEXT:        %[[VAL_8:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[WGSIZE1:.*]] = sycl.range.get %[[VAL_3]]{{\[}}%[[VAL_8]]] : (memref<1x!sycl_range_3_1>, i32) -> index
// CHECK-NEXT:        %[[VAL_11:.*]] = arith.constant 2 : i32
// CHECK-NEXT:        %[[WGSIZE2:.*]] = sycl.range.get %[[VAL_3]]{{\[}}%[[VAL_11]]] : (memref<1x!sycl_range_3_1>, i32) -> index

// COM: Get shared local memory required:
// CHECK-NEXT:        %[[VAL_14:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_15:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_16:.*]] = arith.constant 4 : index
// CHECK-NEXT:        %[[VAL_17:.*]] = arith.muli %[[VAL_16]], %[[WGSIZE0]] : index
// CHECK-NEXT:        %[[VAL_18:.*]] = arith.muli %[[VAL_17]], %[[WGSIZE1]] : index
// CHECK-NEXT:        %[[VAL_19:.*]] = arith.muli %[[VAL_18]], %[[WGSIZE2]] : index
// CHECK-NEXT:        %[[VAL_20:.*]] = arith.addi %[[VAL_15]], %[[VAL_19]] : index
// CHECK-NEXT:        %[[REQD_SHARED_MEM:.*]] = arith.maxsi %[[VAL_14]], %[[VAL_20]] : index

// COM: Get local ids:
// CHECK-NEXT:        %[[VAL_22:.*]] = sycl.local_id : !sycl_id_3_1
// CHECK-NEXT:        %[[VAL_23:.*]] = memref.alloca() : memref<1x!sycl_id_3_1>
// CHECK-NEXT:        %[[VAL_24:.*]] = arith.constant 0 : index
// CHECK-NEXT:        memref.store %[[VAL_22]], %[[VAL_23]]{{\[}}%[[VAL_24]]] : memref<1x!sycl_id_3_1>
// CHECK-NEXT:        %[[VAL_25:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[LOCALID0:.*]] = sycl.id.get %[[VAL_23]]{{\[}}%[[VAL_25]]] : (memref<1x!sycl_id_3_1>, i32) -> index
// CHECK-NEXT:        %[[VAL_28:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[LOCALID1:.*]] = sycl.id.get %[[VAL_23]]{{\[}}%[[VAL_28]]] : (memref<1x!sycl_id_3_1>, i32) -> index
// CHECK-NEXT:        %[[VAL_31:.*]] = arith.constant 2 : i32
// CHECK-NEXT:        %[[LOCALID2:.*]] = sycl.id.get %[[VAL_23]]{{\[}}%[[VAL_31]]] : (memref<1x!sycl_id_3_1>, i32) -> index

// COM: Original code:
// CHECK-NEXT:        %[[VAL_34:.*]] = memref.alloca() : memref<1x!sycl_id_3_>
// CHECK-NEXT:        %[[VAL_35:.*]] = memref.cast %[[VAL_34]] : memref<1x!sycl_id_3_> to memref<?x!sycl_id_3_>
// CHECK-NEXT:        %[[VAL_36:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_37:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_38:.*]] = arith.constant 256 : index
// CHECK-NEXT:        %[[VAL_39:.*]] = arith.constant 512 : index
// CHECK-NEXT:        %[[VAL_40:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[VAL_41:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_42:.*]] = arith.constant 2 : i32
// CHECK-NEXT:        %[[VAL_43:.*]] = sycl.nd_item.get_global_id(%[[VAL_1]], %[[VAL_40]]) : (memref<?x!sycl_nd_item_3_>, i32) -> i64
// CHECK-NEXT:        %[[VAL_44:.*]] = arith.index_cast %[[VAL_43]] : i64 to index
// CHECK-NEXT:        scf.for %[[VAL_45:.*]] = %[[VAL_36]] to %[[VAL_38]] step %[[VAL_37]] {

// COM: Ensure there is a sufficient amount of shared local memory available:
// CHECK-NEXT:          %[[SHARED_MEM_AMOUNT:.*]] = arith.constant 32000 : index
// CHECK-NEXT:          %[[VER_COND:.*]] = arith.cmpi ule, %[[REQD_SHARED_MEM]], %[[SHARED_MEM_AMOUNT]] : index
// CHECK-NEXT:          scf.if %[[VER_COND]] {

// COM: Get pointer to sahred local memory:
// CHECK-NEXT:          %[[VAL_46:.*]] = memref.get_global @WGLocalMem : memref<32000xi8, #sycl.access.address_space<local>>

// COM: Use work group size of dimension 2 as tile size:
// CHECK-NEXT:          %[[TILESIZE:.*]] = arith.muli %[[VAL_37]], %[[WGSIZE2]] : index
// CHECK-NEXT:          scf.for %[[VAL_48:.*]] = %[[VAL_37]] to %[[VAL_39]] step %[[TILESIZE]] {

// COM: Calculate indexes for global memory:
// CHECK-NEXT:            %[[VAL_49:.*]] = arith.addi %[[LOCALID2]], %[[VAL_48]] : index
// CHECK-NEXT:            %[[VAL_50:.*]] = arith.index_cast %[[VAL_49]] : index to i64
// CHECK-NEXT:            %[[VAL_51:.*]] = arith.constant 0 : index

// COM: Get pointer to the shared local memory portion for 1st memref:
// CHECK-NEXT:            %[[VAL_52:.*]] = memref.view %[[VAL_46]]{{\[}}%[[VAL_51]]]{{\[}}%[[WGSIZE0]], %[[WGSIZE1]], %[[WGSIZE2]]] : memref<32000xi8, #sycl.access.address_space<local>> to memref<?x?x?xf32, #sycl.access.address_space<local>>

// COM: Calculate upper bound for the tiled loop:
// CHECK-NEXT:            %[[VAL_53:.*]] = arith.addi %[[VAL_48]], %[[TILESIZE]] : index
// CHECK-NEXT:            %[[VAL_54:.*]] = arith.cmpi slt, %[[VAL_39]], %[[VAL_53]] : index
// CHECK-NEXT:            %[[VAL_55:.*]] = arith.select %[[VAL_54]], %[[VAL_39]], %[[VAL_53]] : index

// COM: Copy to shared local memory for 1st memref:
// CHECK-NEXT:            %[[VAL_56:.*]] = arith.index_cast %[[VAL_45]] : index to i64
// CHECK-NEXT:            %[[VAL_57:.*]] = sycl.id.constructor(%[[VAL_44]], %[[VAL_45]], %[[VAL_49]]) : (index, index, index) -> memref<1x!sycl_id_3_>
// CHECK-NEXT:            %[[VAL_58:.*]] = arith.constant 0 : index
// CHECK-NEXT:            %[[VAL_59:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_57]]] : (memref<?x!sycl_accessor_3_f32_r_dev>, memref<1x!sycl_id_3_>) -> memref<?xf32, 1>
// CHECK-NEXT:            %[[VAL_60:.*]] = memref.load %[[VAL_59]]{{\[}}%[[VAL_58]]] : memref<?xf32, 1>
// CHECK-NEXT:            memref.store %[[VAL_60]], %[[VAL_52]]{{\[}}%[[LOCALID0]], %[[LOCALID1]], %[[LOCALID2]]] : memref<?x?x?xf32, #sycl.access.address_space<local>>

// CHECK-NEXT:            spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:            scf.for %[[VAL_68:.*]] = %[[VAL_48]] to %[[VAL_55]] step %[[VAL_37]] {
// CHECK-NEXT:              %[[VAL_69:.*]] = arith.subi %[[VAL_68]], %[[VAL_48]] : index
// CHECK-NEXT:              %[[VAL_70:.*]] = arith.index_cast %[[VAL_68]] : index to i64
// CHECK-NEXT:              %[[VAL_71:.*]] = arith.index_cast %[[VAL_69]] : index to i64
// CHECK-NEXT:              sycl.constructor @id(%[[VAL_35]], %[[VAL_43]], %[[VAL_56]], %[[VAL_70]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_3_>, i64, i64, i64)
// CHECK-NEXT:              %[[VAL_72:.*]] = memref.load %[[VAL_52]]{{\[}}%[[LOCALID0]], %[[LOCALID1]], %[[VAL_69]]] : memref<?x?x?xf32, #sycl.access.address_space<local>>
// CHECK-NEXT:            }
// CHECK-NEXT:            spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:          }
// CHECK-NEXT:        } else {
// CHECK-NEXT:          scf.for %[[VAL_75:.*]] = %[[VAL_37]] to %[[VAL_39]] step %[[VAL_37]] {
// CHECK-NEXT:            %[[VAL_76:.*]] = arith.index_cast %[[VAL_45]] : index to i64
// CHECK-NEXT:            %[[VAL_77:.*]] = arith.index_cast %[[VAL_75]] : index to i64
// CHECK-NEXT:            sycl.constructor @id(%[[VAL_35]], %[[VAL_43]], %[[VAL_76]], %[[VAL_77]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_3_>, i64, i64, i64)
// CHECK-NEXT:            %[[VAL_78:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_35]]] : (memref<?x!sycl_accessor_3_f32_r_dev>, memref<?x!sycl_id_3_>) -> memref<?xf32>
// CHECK-NEXT:            %[[VAL_79:.*]] = affine.load %[[VAL_78]][0] : memref<?xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      return
gpu.module @device_func {
func.func private @scf_3d(%arg0: memref<?x!sycl_accessor_3_f32_r_dev>, %arg1: memref<?x!sycl_nd_item_3>) {
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
      %subscr1 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_3_f32_r_dev>, memref<?x!sycl_id_3>) -> memref<?xf32>
      %load1 = affine.load %subscr1[0] : memref<?xf32>
    }
  }
  return
}
gpu.func @kernel(%arg0: memref<?x!sycl_accessor_3_f32_r_dev>, %arg1: memref<?x!sycl_nd_item_3>) kernel {
  func.call @scf_3d(%arg0, %arg1) : (memref<?x!sycl_accessor_3_f32_r_dev>, memref<?x!sycl_nd_item_3>) -> ()
  gpu.return
}
}
