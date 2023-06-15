// RUN: polygeist-opt --loop-internalization --split-input-file %s | FileCheck %s

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
// CHECK:       memref.global "private" @WGSharedMem : memref<64xi8, #sycl.access.address_space<local>>
// CHECK-LABEL: func.func private @affine_2d(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<?x!sycl_accessor_2_f32_r_gb>, %[[VAL_1:.*]]: memref<?x!sycl_nd_item_2_>) {

// COM: Get local ids:
// CHECK-NEXT:    %[[VAL_2:.*]] = sycl.local_id : !sycl_id_2_1
// CHECK-NEXT:    %[[VAL_3:.*]] = memref.alloca() : memref<1x!sycl_id_2_1>
// CHECK-NEXT:    %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:    memref.store %[[VAL_2]], %[[VAL_3]]{{\[}}%[[VAL_4]]] : memref<1x!sycl_id_2_1>
// CHECK-NEXT:    %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[VAL_6:.*]] = sycl.id.get %[[VAL_3]]{{\[}}%[[VAL_5]]] : (memref<1x!sycl_id_2_1>, i32) -> memref<?xindex>
// CHECK-NEXT:    %[[VAL_7:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK-NEXT:    %[[VAL_8:.*]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[VAL_9:.*]] = sycl.id.get %[[VAL_3]]{{\[}}%[[VAL_8]]] : (memref<1x!sycl_id_2_1>, i32) -> memref<?xindex>
// CHECK-NEXT:    %[[WGSIZE1:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_4]]] : memref<?xindex>

// COM: Original code:
// CHECK-NEXT:    %[[VAL_11:.*]] = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT:    %[[VAL_12:.*]] = memref.cast %[[VAL_11]] : memref<1x!sycl_id_2_> to memref<?x!sycl_id_2_>
// CHECK-NEXT:    %[[VAL_13:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[VAL_14:.*]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[VAL_15:.*]] = sycl.nd_item.get_global_id(%[[VAL_1]], %[[VAL_13]]) : (memref<?x!sycl_nd_item_2_>, i32) -> i64
// CHECK-NEXT:    %[[VAL_16:.*]] = arith.index_cast %[[VAL_15]] : i64 to index

// COM: Get pointer to shared local memory:
// CHECK-NEXT:    %[[VAL_17:.*]] = memref.get_global @WGSharedMem : memref<64xi8, #sycl.access.address_space<local>>

// COM: Use work group size of dimension 1 as tile size:
// CHECK-NEXT:    %[[TILESIZE:.*]] = arith.constant 4 : index
// CHECK-NEXT:    affine.for %[[VAL_19:.*]] = 0 to [[MAP1]](){{\[}}%[[TILESIZE]]] {

// COM: Get pointer to the shared local memory portion for 1st memref:
// CHECK-NEXT:      %[[VAL_20:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_21:.*]] = memref.view %[[VAL_17]]{{\[}}%[[VAL_20]]][] : memref<64xi8, #sycl.access.address_space<local>> to memref<2x4xf32, #sycl.access.address_space<local>>

// COM: Get tiled loop lower bound:
// CHECK-NEXT:      %[[VAL_22:.*]] = affine.apply [[MAP2]](%[[VAL_19]]){{\[}}%[[TILESIZE]]]

// COM: Calculate indexes for global memory:
// CHECK-NEXT:      %[[VAL_23:.*]] = arith.addi %[[WGSIZE1]], %[[VAL_22]] : index
// CHECK-NEXT:      %[[VAL_24:.*]] = arith.index_cast %[[VAL_23]] : index to i64
// CHECK-NEXT:      %[[VAL_25:.*]] = arith.addi %[[WGSIZE1]], %[[VAL_22]] : index
// CHECK-NEXT:      %[[VAL_26:.*]] = arith.index_cast %[[VAL_25]] : index to i64

// COM: Copy to shared local memory for 1st memref:
// CHECK-NEXT:      %[[VAL_27:.*]] = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT:      %[[VAL_28:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_29:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[VAL_30:.*]] = sycl.id.get %[[VAL_27]]{{\[}}%[[VAL_29]]] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
// CHECK-NEXT:      memref.store %[[VAL_16]], %[[VAL_30]]{{\[}}%[[VAL_28]]] : memref<?xindex>
// CHECK-NEXT:      %[[VAL_31:.*]] = arith.constant 1 : i32
// CHECK-NEXT:      %[[VAL_32:.*]] = sycl.id.get %[[VAL_27]]{{\[}}%[[VAL_31]]] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
// CHECK-NEXT:      memref.store %[[VAL_25]], %[[VAL_32]]{{\[}}%[[VAL_28]]] : memref<?xindex>
// CHECK-NEXT:      %[[VAL_33:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_34:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_27]]] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<1x!sycl_id_2_>) -> memref<?xf32, 1>
// CHECK-NEXT:      %[[VAL_35:.*]] = memref.load %[[VAL_34]]{{\[}}%[[VAL_33]]] : memref<?xf32, 1>
// CHECK-NEXT:      memref.store %[[VAL_35]], %[[VAL_21]]{{\[}}%[[VAL_7]], %[[WGSIZE1]]] : memref<2x4xf32, #sycl.access.address_space<local>>

// COM: Get pointer to the shared local memory portion for 2nd memref:
// CHECK-NEXT:      %[[VAL_36:.*]] = arith.constant 32 : index
// CHECK-NEXT:      %[[VAL_37:.*]] = memref.view %[[VAL_17]]{{\[}}%[[VAL_36]]][] : memref<64xi8, #sycl.access.address_space<local>> to memref<2x4xf32, #sycl.access.address_space<local>>

// COM: Copy to shared local memory for 2nd memref:
// CHECK-NEXT:      %[[VAL_38:.*]] = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT:      %[[VAL_39:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_40:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[VAL_41:.*]] = sycl.id.get %[[VAL_38]]{{\[}}%[[VAL_40]]] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
// CHECK-NEXT:      memref.store %[[VAL_16]], %[[VAL_41]]{{\[}}%[[VAL_39]]] : memref<?xindex>
// CHECK-NEXT:      %[[VAL_42:.*]] = arith.constant 1 : i32
// CHECK-NEXT:      %[[VAL_43:.*]] = sycl.id.get %[[VAL_38]]{{\[}}%[[VAL_42]]] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
// CHECK-NEXT:      memref.store %[[VAL_23]], %[[VAL_43]]{{\[}}%[[VAL_39]]] : memref<?xindex>
// CHECK-NEXT:      %[[VAL_44:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_45:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_38]]] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<1x!sycl_id_2_>) -> memref<?xf32, 1>
// CHECK-NEXT:      %[[VAL_46:.*]] = memref.load %[[VAL_45]]{{\[}}%[[VAL_44]]] : memref<?xf32, 1>
// CHECK-NEXT:      memref.store %[[VAL_46]], %[[VAL_37]]{{\[}}%[[VAL_7]], %[[WGSIZE1]]] : memref<2x4xf32, #sycl.access.address_space<local>>

// CHECK-NEXT:      spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:      affine.for %[[VAL_47:.*]] = [[MAP2]](%[[VAL_19]]){{\[}}%[[TILESIZE]]] to min [[MAP3]](%[[VAL_19]]){{\[}}%[[TILESIZE]]] {
// CHECK-NEXT:        %[[VAL_48:.*]] = arith.subi %[[VAL_47]], %[[VAL_22]] : index
// CHECK-NEXT:        %[[VAL_49:.*]] = arith.index_cast %[[VAL_47]] : index to i64
// CHECK-NEXT:        %[[VAL_50:.*]] = arith.index_cast %[[VAL_48]] : index to i64
// CHECK-NEXT:        %[[VAL_51:.*]] = arith.index_cast %[[VAL_48]] : index to i64
// COM: TODO: sycl.constructor can be removed.
// CHECK-NEXT:        sycl.constructor @id(%[[VAL_12]], %[[VAL_15]], %[[VAL_49]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2_>, i64, i64)
// CHECK-DAG:         %[[VAL_52:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_7]], %[[VAL_48]]] : memref<2x4xf32, #sycl.access.address_space<local>>
// CHECK-DAG:         %[[VAL_53:.*]] = memref.load %[[VAL_37]]{{\[}}%[[VAL_7]], %[[VAL_48]]] : memref<2x4xf32, #sycl.access.address_space<local>>
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
// CHECK:       memref.global "private" @WGSharedMem : memref<32000xi8, #sycl.access.address_space<local>>
// CHECK-LABEL:  func.func private @affine_3d(
// CHECK-SAME:       %[[VAL_0:.*]]: memref<?x!sycl_accessor_3_f32_r_gb>, %[[VAL_1:.*]]: memref<?x!sycl_nd_item_3_>) {

// COM: Get work group sizes:
// CHECK-NEXT:    %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[VAL_3:.*]] = sycl.nd_item.get_local_range(%[[VAL_1]], %[[VAL_2]]) : (memref<?x!sycl_nd_item_3_>, i32) -> i64
// CHECK-NEXT:    %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]] : i64 to index
// CHECK-NEXT:    %[[VAL_5:.*]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[VAL_6:.*]] = sycl.nd_item.get_local_range(%[[VAL_1]], %[[VAL_5]]) : (memref<?x!sycl_nd_item_3_>, i32) -> i64
// CHECK-NEXT:    %[[VAL_7:.*]] = arith.index_cast %[[VAL_6]] : i64 to index
// CHECK-NEXT:    %[[VAL_8:.*]] = arith.constant 2 : i32
// CHECK-NEXT:    %[[VAL_9:.*]] = sycl.nd_item.get_local_range(%[[VAL_1]], %[[VAL_8]]) : (memref<?x!sycl_nd_item_3_>, i32) -> i64
// CHECK-NEXT:    %[[WGSIZE2:.*]] = arith.index_cast %[[VAL_9]] : i64 to index

// COM: Get shared local memory required:
// COM:   for each loop, get the max of:
// COM:     for each accessor subscript, add:
// COM:       multiply element size (4) with work group size for each dimension
// CHECK-NEXT:    %[[VAL_11:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[VAL_12:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[VAL_13:.*]] = arith.constant 4 : index
// CHECK-NEXT:    %[[VAL_14:.*]] = arith.muli %[[VAL_13]], %[[VAL_4]] : index
// CHECK-NEXT:    %[[VAL_15:.*]] = arith.muli %[[VAL_14]], %[[VAL_7]] : index
// CHECK-NEXT:    %[[VAL_16:.*]] = arith.muli %[[VAL_15]], %[[WGSIZE2]] : index
// CHECK-NEXT:    %[[VAL_17:.*]] = arith.addi %[[VAL_12]], %[[VAL_16]] : index
// CHECK-NEXT:    %[[VAL_18:.*]] = arith.maxsi %[[VAL_11]], %[[VAL_17]] : index

// COM: Get local ids:
// CHECK-NEXT:    %[[VAL_19:.*]] = sycl.local_id : !sycl_id_3_1
// CHECK-NEXT:    %[[VAL_20:.*]] = memref.alloca() : memref<1x!sycl_id_3_1>
// CHECK-NEXT:    %[[VAL_21:.*]] = arith.constant 0 : index
// CHECK-NEXT:    memref.store %[[VAL_19]], %[[VAL_20]]{{\[}}%[[VAL_21]]] : memref<1x!sycl_id_3_1>
// CHECK-NEXT:    %[[VAL_22:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[VAL_23:.*]] = sycl.id.get %[[VAL_20]]{{\[}}%[[VAL_22]]] : (memref<1x!sycl_id_3_1>, i32) -> memref<?xindex>
// CHECK-NEXT:    %[[VAL_24:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_21]]] : memref<?xindex>
// CHECK-NEXT:    %[[VAL_25:.*]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[VAL_26:.*]] = sycl.id.get %[[VAL_20]]{{\[}}%[[VAL_25]]] : (memref<1x!sycl_id_3_1>, i32) -> memref<?xindex>
// CHECK-NEXT:    %[[VAL_27:.*]] = memref.load %[[VAL_26]]{{\[}}%[[VAL_21]]] : memref<?xindex>
// CHECK-NEXT:    %[[VAL_28:.*]] = arith.constant 2 : i32
// CHECK-NEXT:    %[[VAL_29:.*]] = sycl.id.get %[[VAL_20]]{{\[}}%[[VAL_28]]] : (memref<1x!sycl_id_3_1>, i32) -> memref<?xindex>
// CHECK-NEXT:    %[[VAL_30:.*]] = memref.load %[[VAL_29]]{{\[}}%[[VAL_21]]] : memref<?xindex>

// COM: Original code:
// CHECK-NEXT:    %[[VAL_31:.*]] = memref.alloca() : memref<1x!sycl_id_3_>
// CHECK-NEXT:    %[[VAL_32:.*]] = memref.cast %[[VAL_31]] : memref<1x!sycl_id_3_> to memref<?x!sycl_id_3_>
// CHECK-NEXT:    %[[VAL_33:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[VAL_34:.*]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[VAL_35:.*]] = arith.constant 2 : i32
// CHECK-NEXT:    %[[VAL_36:.*]] = sycl.nd_item.get_global_id(%[[VAL_1]], %[[VAL_33]]) : (memref<?x!sycl_nd_item_3_>, i32) -> i64
// CHECK-NEXT:    %[[VAL_37:.*]] = arith.index_cast %[[VAL_36]] : i64 to index
// CHECK-NEXT:    %[[VAL_38:.*]] = sycl.nd_item.get_global_id(%[[VAL_1]], %[[VAL_34]]) : (memref<?x!sycl_nd_item_3_>, i32) -> i64
// CHECK-NEXT:    %[[VAL_39:.*]] = sycl.nd_item.get_global_id(%[[VAL_1]], %[[VAL_35]]) : (memref<?x!sycl_nd_item_3_>, i32) -> i64
// CHECK-NEXT:    affine.for %[[VAL_40:.*]] = 0 to 256 {

// COM: Get pointer to shared local memory:
// CHECK-NEXT:      %[[VAL_41:.*]] = memref.get_global @WGSharedMem : memref<32000xi8, #sycl.access.address_space<local>>

// COM: Use work group size of dimension 2 as tile size:
// CHECK-NEXT:      affine.for %[[VAL_42:.*]] = 1 to [[MAP1]](){{\[}}%[[WGSIZE2]]] {

// COM: Get pointer to the shared local memory portion for 1st memref:
// CHECK-NEXT:        %[[VAL_43:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_44:.*]] = memref.view %[[VAL_41]]{{\[}}%[[VAL_43]]]{{\[}}%[[VAL_4]], %[[VAL_7]], %[[WGSIZE2]]] : memref<32000xi8, #sycl.access.address_space<local>> to memref<?x?x?xf32, #sycl.access.address_space<local>>

// COM: Calculate indexes for global memory:
// CHECK-NEXT:        %[[VAL_45:.*]] = arith.index_cast %[[VAL_40]] : index to i64
// CHECK-NEXT:        %[[VAL_46:.*]] = affine.apply [[MAP2]](%[[VAL_42]]){{\[}}%[[WGSIZE2]]]
// CHECK-NEXT:        %[[VAL_47:.*]] = arith.addi %[[VAL_30]], %[[VAL_46]] : index
// CHECK-NEXT:        %[[VAL_48:.*]] = arith.index_cast %[[VAL_47]] : index to i64

// COM: Copy to shared local memory for 1st memref:
// CHECK-NEXT:        %[[VAL_49:.*]] = memref.alloca() : memref<1x!sycl_id_3_>
// CHECK-NEXT:        %[[VAL_50:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_51:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[VAL_52:.*]] = sycl.id.get %[[VAL_49]]{{\[}}%[[VAL_51]]] : (memref<1x!sycl_id_3_>, i32) -> memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_37]], %[[VAL_52]]{{\[}}%[[VAL_50]]] : memref<?xindex>
// CHECK-NEXT:        %[[VAL_53:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_54:.*]] = sycl.id.get %[[VAL_49]]{{\[}}%[[VAL_53]]] : (memref<1x!sycl_id_3_>, i32) -> memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_40]], %[[VAL_54]]{{\[}}%[[VAL_50]]] : memref<?xindex>
// CHECK-NEXT:        %[[VAL_55:.*]] = arith.constant 2 : i32
// CHECK-NEXT:        %[[VAL_56:.*]] = sycl.id.get %[[VAL_49]]{{\[}}%[[VAL_55]]] : (memref<1x!sycl_id_3_>, i32) -> memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_47]], %[[VAL_56]]{{\[}}%[[VAL_50]]] : memref<?xindex>
// CHECK-NEXT:        %[[VAL_57:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_58:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_49]]] : (memref<?x!sycl_accessor_3_f32_r_gb>, memref<1x!sycl_id_3_>) -> memref<?xf32, 1>
// CHECK-NEXT:        %[[VAL_59:.*]] = memref.load %[[VAL_58]]{{\[}}%[[VAL_57]]] : memref<?xf32, 1>
// CHECK-NEXT:        memref.store %[[VAL_59]], %[[VAL_44]]{{\[}}%[[VAL_24]], %[[VAL_27]], %[[VAL_30]]] : memref<?x?x?xf32, #sycl.access.address_space<local>>

// CHECK-NEXT:        spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:        affine.for %[[VAL_60:.*]] = [[MAP2]](%[[VAL_42]]){{\[}}%[[WGSIZE2]]] to min [[MAP3]](%[[VAL_42]]){{\[}}%[[WGSIZE2]]] {
// CHECK-NEXT:          %[[VAL_61:.*]] = arith.subi %[[VAL_60]], %[[VAL_46]] : index
// CHECK-NEXT:          %[[VAL_62:.*]] = arith.index_cast %[[VAL_60]] : index to i64
// CHECK-NEXT:          %[[VAL_63:.*]] = arith.index_cast %[[VAL_61]] : index to i64
// CHECK-NEXT:          sycl.constructor @id(%[[VAL_32]], %[[VAL_36]], %[[VAL_45]], %[[VAL_62]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_3_>, i64, i64, i64)
// CHECK-NEXT:          %[[VAL_64:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_24]], %[[VAL_27]], %[[VAL_61]]] : memref<?x?x?xf32, #sycl.access.address_space<local>>
// CHECK-NEXT:          sycl.constructor @id(%[[VAL_32]], %[[VAL_36]], %[[VAL_38]], %[[VAL_39]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_3_>, i64, i64, i64)
// CHECK-NEXT:          %[[VAL_65:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_32]]] : (memref<?x!sycl_accessor_3_f32_r_gb>, memref<?x!sycl_id_3_>) -> memref<?xf32>
// CHECK-NEXT:          %[[VAL_66:.*]] = affine.load %[[VAL_65]][0] : memref<?xf32>
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
  %ty = sycl.nd_item.get_global_id(%arg1, %c1_i32) : (memref<?x!sycl_nd_item_3>, i32) -> i64
  %tz = sycl.nd_item.get_global_id(%arg1, %c2_i32) : (memref<?x!sycl_nd_item_3>, i32) -> i64

  affine.for %ii = 0 to 256 {
    affine.for %jj = 1 to 512 {
      %i = arith.index_cast %ii : index to i64
      %j = arith.index_cast %jj : index to i64

      // Should use shared memory (access exhibits temporal locality).
      sycl.constructor @id(%id, %tx, %i, %j) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_3>, i64, i64, i64)
      %subscr1 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_3_f32_r_gb>, memref<?x!sycl_id_3>) -> memref<?xf32>
      %load1 = affine.load %subscr1[0] : memref<?xf32>

      // Should use global memory (access is coalesced).
      sycl.constructor @id(%id, %tx, %ty, %tz) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_3>, i64, i64, i64)
      %subscr2 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_3_f32_r_gb>, memref<?x!sycl_id_3>) -> memref<?xf32>      
      %load2 = affine.load %subscr2[0] : memref<?xf32>      
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

// CHECK:           memref.global "private" @WGSharedMem : memref<32000xi8, #sycl.access.address_space<local>>
// CHECK-LABEL:     func.func private @scf_2d(
// CHECK-SAME:          %[[VAL_0:.*]]: memref<?x!sycl_accessor_2_f32_r_gb>, %[[VAL_1:.*]]: memref<?x!sycl_nd_item_2_>) {

// COM: Get work group sizes:
// CHECK-NEXT:        %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[VAL_3:.*]] = sycl.nd_item.get_local_range(%[[VAL_1]], %[[VAL_2]]) : (memref<?x!sycl_nd_item_2_>, i32) -> i64
// CHECK-NEXT:        %[[WGSIZE0:.*]] = arith.index_cast %[[VAL_3]] : i64 to index
// CHECK-NEXT:        %[[VAL_5:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_6:.*]] = sycl.nd_item.get_local_range(%[[VAL_1]], %[[VAL_5]]) : (memref<?x!sycl_nd_item_2_>, i32) -> i64
// CHECK-NEXT:        %[[WGSIZE1:.*]] = arith.index_cast %[[VAL_6]] : i64 to index

// COM: Get shared local memory required:
// CHECK-NEXT:        %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_9:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_10:.*]] = arith.constant 4 : index
// CHECK-NEXT:        %[[VAL_11:.*]] = arith.muli %[[VAL_10]], %[[WGSIZE0]] : index
// CHECK-NEXT:        %[[VAL_12:.*]] = arith.muli %[[VAL_11]], %[[WGSIZE1]] : index
// CHECK-NEXT:        %[[VAL_13:.*]] = arith.addi %[[VAL_9]], %[[VAL_12]] : index
// CHECK-NEXT:        %[[VAL_14:.*]] = arith.constant 4 : index
// CHECK-NEXT:        %[[VAL_15:.*]] = arith.muli %[[VAL_14]], %[[WGSIZE0]] : index
// CHECK-NEXT:        %[[VAL_16:.*]] = arith.muli %[[VAL_15]], %[[WGSIZE1]] : index
// CHECK-NEXT:        %[[VAL_17:.*]] = arith.addi %[[VAL_13]], %[[VAL_16]] : index
// CHECK-NEXT:        %[[VAL_18:.*]] = arith.maxsi %[[VAL_8]], %[[VAL_17]] : index

// COM: Get local ids:
// CHECK-NEXT:        %[[VAL_19:.*]] = sycl.local_id : !sycl_id_2_1
// CHECK-NEXT:        %[[VAL_20:.*]] = memref.alloca() : memref<1x!sycl_id_2_1>
// CHECK-NEXT:        %[[VAL_21:.*]] = arith.constant 0 : index
// CHECK-NEXT:        memref.store %[[VAL_19]], %[[VAL_20]]{{\[}}%[[VAL_21]]] : memref<1x!sycl_id_2_1>
// CHECK-NEXT:        %[[VAL_22:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[VAL_23:.*]] = sycl.id.get %[[VAL_20]]{{\[}}%[[VAL_22]]] : (memref<1x!sycl_id_2_1>, i32) -> memref<?xindex>
// CHECK-NEXT:        %[[VAL_24:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_21]]] : memref<?xindex>
// CHECK-NEXT:        %[[VAL_25:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_26:.*]] = sycl.id.get %[[VAL_20]]{{\[}}%[[VAL_25]]] : (memref<1x!sycl_id_2_1>, i32) -> memref<?xindex>
// CHECK-NEXT:        %[[VAL_27:.*]] = memref.load %[[VAL_26]]{{\[}}%[[VAL_21]]] : memref<?xindex>

// COM: Original code:
// CHECK-NEXT:        %[[VAL_28:.*]] = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT:        %[[VAL_29:.*]] = memref.cast %[[VAL_28]] : memref<1x!sycl_id_2_> to memref<?x!sycl_id_2_>
// CHECK-NEXT:        %[[VAL_30:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_31:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_32:.*]] = arith.constant 256 : index
// CHECK-NEXT:        %[[VAL_33:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[VAL_34:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_35:.*]] = sycl.nd_item.get_global_id(%[[VAL_1]], %[[VAL_33]]) : (memref<?x!sycl_nd_item_2_>, i32) -> i64
// CHECK-NEXT:        %[[VAL_36:.*]] = arith.index_cast %[[VAL_35]] : i64 to index

// CHECK-NEXT:        %[[VER_COND:.*]] = arith.cmpi eq, %[[WGSIZE0]], %[[WGSIZE1]] : index
// CHECK-NEXT:        scf.if %[[VER_COND]] {

// COM: Get pointer to shared local memory:
// CHECK-NEXT:        %[[VAL_37:.*]] = memref.get_global @WGSharedMem : memref<32000xi8, #sycl.access.address_space<local>>

// COM: Use work group size as tile size:
// CHECK-NEXT:        %[[TILESIZE:.*]] = arith.muli %[[VAL_31]], %[[WGSIZE0]] : index
// CHECK-NEXT:        scf.for %[[VAL_39:.*]] = %[[VAL_30]] to %[[VAL_32]] step %[[TILESIZE]] {

// COM: Calculate indexes for global memory:
// CHECK-NEXT:          %[[VAL_40:.*]] = arith.addi %[[VAL_24]], %[[VAL_39]] : index
// CHECK-NEXT:          %[[VAL_41:.*]] = arith.index_cast %[[VAL_40]] : index to i64
// CHECK-NEXT:          %[[VAL_42:.*]] = arith.addi %[[VAL_27]], %[[VAL_39]] : index
// CHECK-NEXT:          %[[VAL_43:.*]] = arith.index_cast %[[VAL_42]] : index to i64
// CHECK-NEXT:          %[[VAL_44:.*]] = arith.constant 0 : index

// COM: Get pointer to the shared local memory portion for 1st memref:
// CHECK-NEXT:          %[[VAL_45:.*]] = memref.view %[[VAL_37]]{{\[}}%[[VAL_44]]]{{\[}}%[[WGSIZE0]], %[[WGSIZE1]]] : memref<32000xi8, #sycl.access.address_space<local>> to memref<?x?xf32, #sycl.access.address_space<local>>

// COM: Calculate upper bound for the tiled loop:
// CHECK-NEXT:          %[[VAL_46:.*]] = arith.addi %[[VAL_39]], %[[TILESIZE]] : index
// CHECK-NEXT:          %[[VAL_47:.*]] = arith.cmpi slt, %[[VAL_32]], %[[VAL_46]] : index
// CHECK-NEXT:          %[[VAL_48:.*]] = arith.select %[[VAL_47]], %[[VAL_32]], %[[VAL_46]] : index

// COM: Copy to shared local memory for 1st memref:
// CHECK-NEXT:          %[[VAL_49:.*]] = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT:          %[[VAL_50:.*]] = arith.constant 0 : index
// CHECK-NEXT:          %[[VAL_51:.*]] = arith.constant 0 : i32
// CHECK-NEXT:          %[[VAL_52:.*]] = sycl.id.get %[[VAL_49]]{{\[}}%[[VAL_51]]] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
// CHECK-NEXT:          memref.store %[[VAL_36]], %[[VAL_52]]{{\[}}%[[VAL_50]]] : memref<?xindex>
// CHECK-NEXT:          %[[VAL_53:.*]] = arith.constant 1 : i32
// CHECK-NEXT:          %[[VAL_54:.*]] = sycl.id.get %[[VAL_49]]{{\[}}%[[VAL_53]]] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
// CHECK-NEXT:          memref.store %[[VAL_42]], %[[VAL_54]]{{\[}}%[[VAL_50]]] : memref<?xindex>
// CHECK-NEXT:          %[[VAL_55:.*]] = arith.constant 0 : index
// CHECK-NEXT:          %[[VAL_56:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_49]]] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<1x!sycl_id_2_>) -> memref<?xf32, 1>
// CHECK-NEXT:          %[[VAL_57:.*]] = memref.load %[[VAL_56]]{{\[}}%[[VAL_55]]] : memref<?xf32, 1>
// CHECK-NEXT:          memref.store %[[VAL_57]], %[[VAL_45]]{{\[}}%[[VAL_24]], %[[VAL_27]]] : memref<?x?xf32, #sycl.access.address_space<local>>

// COM: Calculate offset:
// CHECK-NEXT:          %[[VAL_58:.*]] = arith.constant 4 : index
// CHECK-NEXT:          %[[VAL_59:.*]] = arith.muli %[[VAL_58]], %[[WGSIZE0]] : index
// CHECK-NEXT:          %[[VAL_60:.*]] = arith.muli %[[VAL_59]], %[[WGSIZE1]] : index
// CHECK-NEXT:          %[[VAL_61:.*]] = arith.addi %[[VAL_44]], %[[VAL_60]] : index

// COM: Get pointer to the shared local memory portion for 2nd memref:
// CHECK-NEXT:          %[[VAL_62:.*]] = memref.view %[[VAL_37]]{{\[}}%[[VAL_61]]]{{\[}}%[[WGSIZE0]], %[[WGSIZE1]]] : memref<32000xi8, #sycl.access.address_space<local>> to memref<?x?xf32, #sycl.access.address_space<local>>

// COM: Copy to shared local memory for 2nd memref:
// CHECK-NEXT:          %[[VAL_63:.*]] = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT:          %[[VAL_64:.*]] = arith.constant 0 : index
// CHECK-NEXT:          %[[VAL_65:.*]] = arith.constant 0 : i32
// CHECK-NEXT:          %[[VAL_66:.*]] = sycl.id.get %[[VAL_63]]{{\[}}%[[VAL_65]]] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
// CHECK-NEXT:          memref.store %[[VAL_40]], %[[VAL_66]]{{\[}}%[[VAL_64]]] : memref<?xindex>
// CHECK-NEXT:          %[[VAL_67:.*]] = arith.constant 1 : i32
// CHECK-NEXT:          %[[VAL_68:.*]] = sycl.id.get %[[VAL_63]]{{\[}}%[[VAL_67]]] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
// CHECK-NEXT:          memref.store %[[VAL_36]], %[[VAL_68]]{{\[}}%[[VAL_64]]] : memref<?xindex>
// CHECK-NEXT:          %[[VAL_69:.*]] = arith.constant 0 : index
// CHECK-NEXT:          %[[VAL_70:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_63]]] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<1x!sycl_id_2_>) -> memref<?xf32, 1>
// CHECK-NEXT:          %[[VAL_71:.*]] = memref.load %[[VAL_70]]{{\[}}%[[VAL_69]]] : memref<?xf32, 1>
// CHECK-NEXT:          memref.store %[[VAL_71]], %[[VAL_62]]{{\[}}%[[VAL_24]], %[[VAL_27]]] : memref<?x?xf32, #sycl.access.address_space<local>>

// CHECK-NEXT:          spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:          scf.for %[[VAL_72:.*]] = %[[VAL_39]] to %[[VAL_48]] step %[[VAL_31]] {
// CHECK-NEXT:            %[[VAL_73:.*]] = arith.subi %[[VAL_72]], %[[VAL_39]] : index
// CHECK-NEXT:            %[[VAL_74:.*]] = arith.index_cast %[[VAL_72]] : index to i64
// CHECK-NEXT:            %[[VAL_75:.*]] = arith.index_cast %[[VAL_73]] : index to i64
// CHECK-NEXT:            %[[VAL_76:.*]] = arith.index_cast %[[VAL_73]] : index to i64
// CHECK-NEXT:            sycl.constructor @id(%[[VAL_29]], %[[VAL_35]], %[[VAL_74]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2_>, i64, i64)
// CHECK-NEXT:            memref.load %[[VAL_45]]{{\[}}%[[VAL_24]], %[[VAL_73]]] : memref<?x?xf32, #sycl.access.address_space<local>>
// CHECK-NEXT:            sycl.constructor @id(%[[VAL_29]], %[[VAL_74]], %[[VAL_35]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2_>, i64, i64)
// CHECK-NEXT:            memref.load %[[VAL_62]]{{\[}}%[[VAL_73]], %[[VAL_27]]] : memref<?x?xf32, #sycl.access.address_space<local>>
// CHECK-NEXT:          }
// CHECK-NEXT:          spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:        }
// CHECK-NEXT:      } else {
// CHECK-NEXT:        scf.for %[[VAL_80:.*]] = %[[VAL_30]] to %[[VAL_32]] step %[[VAL_31]] {
// CHECK-NEXT:          %[[VAL_81:.*]] = arith.index_cast %[[VAL_80]] : index to i64
// CHECK-NEXT:          sycl.constructor @id(%[[VAL_29]], %[[VAL_35]], %[[VAL_81]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2_>, i64, i64)
// CHECK-NEXT:          %[[VAL_82:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_29]]] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<?x!sycl_id_2_>) -> memref<?xf32>
// CHECK-NEXT:          %[[VAL_83:.*]] = affine.load %[[VAL_82]][0] : memref<?xf32>
// CHECK-NEXT:          sycl.constructor @id(%[[VAL_29]], %[[VAL_81]], %[[VAL_35]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2_>, i64, i64)
// CHECK-NEXT:          %[[VAL_84:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_29]]] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<?x!sycl_id_2_>) -> memref<?xf32>
// CHECK-NEXT:          %[[VAL_85:.*]] = affine.load %[[VAL_84]][0] : memref<?xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
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

    sycl.constructor @id(%id, %i, %tx) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2>, i64, i64)    
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

// CHECK:           memref.global "private" @WGSharedMem : memref<32000xi8, #sycl.access.address_space<local>>
// CHECK-LABEL:     func.func private @scf_3d(
// CHECK-SAME:          %[[VAL_0:.*]]: memref<?x!sycl_accessor_3_f32_r_gb>, %[[VAL_1:.*]]: memref<?x!sycl_nd_item_3_>) {

// COM: Get work group sizes:
// CHECK-NEXT:        %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[VAL_3:.*]] = sycl.nd_item.get_local_range(%[[VAL_1]], %[[VAL_2]]) : (memref<?x!sycl_nd_item_3_>, i32) -> i64
// CHECK-NEXT:        %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]] : i64 to index
// CHECK-NEXT:        %[[VAL_5:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_6:.*]] = sycl.nd_item.get_local_range(%[[VAL_1]], %[[VAL_5]]) : (memref<?x!sycl_nd_item_3_>, i32) -> i64
// CHECK-NEXT:        %[[VAL_7:.*]] = arith.index_cast %[[VAL_6]] : i64 to index
// CHECK-NEXT:        %[[VAL_8:.*]] = arith.constant 2 : i32
// CHECK-NEXT:        %[[VAL_9:.*]] = sycl.nd_item.get_local_range(%[[VAL_1]], %[[VAL_8]]) : (memref<?x!sycl_nd_item_3_>, i32) -> i64
// CHECK-NEXT:        %[[WGSIZE2:.*]] = arith.index_cast %[[VAL_9]] : i64 to index

// COM: Get shared local memory required:
// CHECK-NEXT:        %[[VAL_11:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_12:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_13:.*]] = arith.constant 4 : index
// CHECK-NEXT:        %[[VAL_14:.*]] = arith.muli %[[VAL_13]], %[[VAL_4]] : index
// CHECK-NEXT:        %[[VAL_15:.*]] = arith.muli %[[VAL_14]], %[[VAL_7]] : index
// CHECK-NEXT:        %[[VAL_16:.*]] = arith.muli %[[VAL_15]], %[[WGSIZE2]] : index
// CHECK-NEXT:        %[[VAL_17:.*]] = arith.addi %[[VAL_12]], %[[VAL_16]] : index
// CHECK-NEXT:        %[[VAL_18:.*]] = arith.maxsi %[[VAL_11]], %[[VAL_17]] : index

// COM: Get local ids:
// CHECK-NEXT:        %[[VAL_19:.*]] = sycl.local_id : !sycl_id_3_1
// CHECK-NEXT:        %[[VAL_20:.*]] = memref.alloca() : memref<1x!sycl_id_3_1>
// CHECK-NEXT:        %[[VAL_21:.*]] = arith.constant 0 : index
// CHECK-NEXT:        memref.store %[[VAL_19]], %[[VAL_20]]{{\[}}%[[VAL_21]]] : memref<1x!sycl_id_3_1>
// CHECK-NEXT:        %[[VAL_22:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[VAL_23:.*]] = sycl.id.get %[[VAL_20]]{{\[}}%[[VAL_22]]] : (memref<1x!sycl_id_3_1>, i32) -> memref<?xindex>
// CHECK-NEXT:        %[[VAL_24:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_21]]] : memref<?xindex>
// CHECK-NEXT:        %[[VAL_25:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_26:.*]] = sycl.id.get %[[VAL_20]]{{\[}}%[[VAL_25]]] : (memref<1x!sycl_id_3_1>, i32) -> memref<?xindex>
// CHECK-NEXT:        %[[VAL_27:.*]] = memref.load %[[VAL_26]]{{\[}}%[[VAL_21]]] : memref<?xindex>
// CHECK-NEXT:        %[[VAL_28:.*]] = arith.constant 2 : i32
// CHECK-NEXT:        %[[VAL_29:.*]] = sycl.id.get %[[VAL_20]]{{\[}}%[[VAL_28]]] : (memref<1x!sycl_id_3_1>, i32) -> memref<?xindex>
// CHECK-NEXT:        %[[VAL_30:.*]] = memref.load %[[VAL_29]]{{\[}}%[[VAL_21]]] : memref<?xindex>

// COM: Original code:
// CHECK-NEXT:        %[[VAL_31:.*]] = memref.alloca() : memref<1x!sycl_id_3_>
// CHECK-NEXT:        %[[VAL_32:.*]] = memref.cast %[[VAL_31]] : memref<1x!sycl_id_3_> to memref<?x!sycl_id_3_>
// CHECK-NEXT:        %[[VAL_33:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_34:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_35:.*]] = arith.constant 256 : index
// CHECK-NEXT:        %[[VAL_36:.*]] = arith.constant 512 : index
// CHECK-NEXT:        %[[VAL_37:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[VAL_38:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_39:.*]] = arith.constant 2 : i32
// CHECK-NEXT:        %[[VAL_40:.*]] = sycl.nd_item.get_global_id(%[[VAL_1]], %[[VAL_37]]) : (memref<?x!sycl_nd_item_3_>, i32) -> i64
// CHECK-NEXT:        %[[VAL_41:.*]] = arith.index_cast %[[VAL_40]] : i64 to index
// CHECK-NEXT:        scf.for %[[VAL_42:.*]] = %[[VAL_33]] to %[[VAL_35]] step %[[VAL_34]] {

// COM: Get pointer to shared local memory:
// CHECK-NEXT:          %[[VAL_43:.*]] = memref.get_global @WGSharedMem : memref<32000xi8, #sycl.access.address_space<local>>

// COM: Use work group size of dimension 1 as tile size:
// CHECK-NEXT:          %[[VAL_44:.*]] = arith.muli %[[VAL_34]], %[[WGSIZE2]] : index
// CHECK-NEXT:          scf.for %[[VAL_45:.*]] = %[[VAL_34]] to %[[VAL_36]] step %[[VAL_44]] {

// COM: Calculate indexes for global memory:
// CHECK-NEXT:            %[[VAL_46:.*]] = arith.addi %[[VAL_30]], %[[VAL_45]] : index
// CHECK-NEXT:            %[[VAL_47:.*]] = arith.index_cast %[[VAL_46]] : index to i64
// CHECK-NEXT:            %[[VAL_48:.*]] = arith.constant 0 : index

// COM: Get pointer to the shared local memory portion for 1st memref:
// CHECK-NEXT:            %[[VAL_49:.*]] = memref.view %[[VAL_43]]{{\[}}%[[VAL_48]]]{{\[}}%[[VAL_4]], %[[VAL_7]], %[[WGSIZE2]]] : memref<32000xi8, #sycl.access.address_space<local>> to memref<?x?x?xf32, #sycl.access.address_space<local>>

// COM: Calculate upper bound for the tiled loop:
// CHECK-NEXT:            %[[VAL_50:.*]] = arith.addi %[[VAL_45]], %[[VAL_44]] : index
// CHECK-NEXT:            %[[VAL_51:.*]] = arith.cmpi slt, %[[VAL_36]], %[[VAL_50]] : index
// CHECK-NEXT:            %[[VAL_52:.*]] = arith.select %[[VAL_51]], %[[VAL_36]], %[[VAL_50]] : index

// COM: Copy to shared local memory for 1st memref:
// CHECK-NEXT:            %[[VAL_53:.*]] = arith.index_cast %[[VAL_42]] : index to i64
// CHECK-NEXT:            %[[VAL_54:.*]] = memref.alloca() : memref<1x!sycl_id_3_>
// CHECK-NEXT:            %[[VAL_55:.*]] = arith.constant 0 : index
// CHECK-NEXT:            %[[VAL_56:.*]] = arith.constant 0 : i32
// CHECK-NEXT:            %[[VAL_57:.*]] = sycl.id.get %[[VAL_54]]{{\[}}%[[VAL_56]]] : (memref<1x!sycl_id_3_>, i32) -> memref<?xindex>
// CHECK-NEXT:            memref.store %[[VAL_41]], %[[VAL_57]]{{\[}}%[[VAL_55]]] : memref<?xindex>
// CHECK-NEXT:            %[[VAL_58:.*]] = arith.constant 1 : i32
// CHECK-NEXT:            %[[VAL_59:.*]] = sycl.id.get %[[VAL_54]]{{\[}}%[[VAL_58]]] : (memref<1x!sycl_id_3_>, i32) -> memref<?xindex>
// CHECK-NEXT:            memref.store %[[VAL_42]], %[[VAL_59]]{{\[}}%[[VAL_55]]] : memref<?xindex>
// CHECK-NEXT:            %[[VAL_60:.*]] = arith.constant 2 : i32
// CHECK-NEXT:            %[[VAL_61:.*]] = sycl.id.get %[[VAL_54]]{{\[}}%[[VAL_60]]] : (memref<1x!sycl_id_3_>, i32) -> memref<?xindex>
// CHECK-NEXT:            memref.store %[[VAL_46]], %[[VAL_61]]{{\[}}%[[VAL_55]]] : memref<?xindex>
// CHECK-NEXT:            %[[VAL_62:.*]] = arith.constant 0 : index
// CHECK-NEXT:            %[[VAL_63:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_54]]] : (memref<?x!sycl_accessor_3_f32_r_gb>, memref<1x!sycl_id_3_>) -> memref<?xf32, 1>
// CHECK-NEXT:            %[[VAL_64:.*]] = memref.load %[[VAL_63]]{{\[}}%[[VAL_62]]] : memref<?xf32, 1>
// CHECK-NEXT:            memref.store %[[VAL_64]], %[[VAL_49]]{{\[}}%[[VAL_24]], %[[VAL_27]], %[[VAL_30]]] : memref<?x?x?xf32, #sycl.access.address_space<local>>

// CHECK-NEXT:            spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:            scf.for %[[VAL_65:.*]] = %[[VAL_45]] to %[[VAL_52]] step %[[VAL_34]] {
// CHECK-NEXT:              %[[VAL_66:.*]] = arith.subi %[[VAL_65]], %[[VAL_45]] : index
// CHECK-NEXT:              %[[VAL_67:.*]] = arith.index_cast %[[VAL_65]] : index to i64
// CHECK-NEXT:              %[[VAL_68:.*]] = arith.index_cast %[[VAL_66]] : index to i64
// CHECK-NEXT:              sycl.constructor @id(%[[VAL_32]], %[[VAL_40]], %[[VAL_53]], %[[VAL_67]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_3_>, i64, i64, i64)
// CHECK-NEXT:              %[[VAL_69:.*]] = memref.load %[[VAL_49]]{{\[}}%[[VAL_24]], %[[VAL_27]], %[[VAL_66]]] : memref<?x?x?xf32, #sycl.access.address_space<local>>
// CHECK-NEXT:            }
// CHECK-NEXT:            spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        return
// CHECK-NEXT:      }
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
