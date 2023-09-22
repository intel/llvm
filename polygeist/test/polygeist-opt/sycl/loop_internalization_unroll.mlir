// RUN: polygeist-opt --loop-internalization="unroll-factor=2" -mlir-pass-statistics %s 2>&1 | FileCheck %s

// CHECK-DAG:   (S) 1 num-access-internalized - Number of accesses internalized
// CHECK-DAG:   (S) 1 num-loop-internalized - Number of loops internalized

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_item_base_1_ = !sycl.item_base<[2, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_accessor_1_f32_r_dev = !sycl.accessor<[1, f32, read, device], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xf32, 2>)>)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>

// CHECK-DAG:   [[MAP1:#map.*]] = affine_map<()[s0] -> (256 ceildiv s0)>
// CHECK-DAG:   [[MAP2:#map.*]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK:       memref.global "private" @WGLocalMem : memref<32000xi8, #sycl.access.address_space<local>>
// CHECK-LABEL: func.func private @affine(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<?x!sycl_accessor_1_f32_r_dev>, %[[VAL_1:.*]]: memref<?x!sycl_item_1_>) {
// CHECK-NEXT:    %[[VAL_2:.*]] = sycl.work_group_size : !sycl_range_1_1
// CHECK-NEXT:    %[[VAL_3:.*]] = memref.alloca() : memref<1x!sycl_range_1_1>
// CHECK-NEXT:    %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:    memref.store %[[VAL_2]], %[[VAL_3]]{{\[}}%[[VAL_4]]] : memref<1x!sycl_range_1_1>
// CHECK-NEXT:    %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[VAL_6:.*]] = sycl.range.get %[[VAL_3]]{{\[}}%[[VAL_5]]] : (memref<1x!sycl_range_1_1>, i32) -> index
// CHECK-NEXT:    %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[VAL_9:.*]] = arith.constant 4 : index
// CHECK-NEXT:    %[[VAL_10:.*]] = arith.muli %[[VAL_9]], %[[VAL_6]] : index
// CHECK-NEXT:    %[[VAL_11:.*]] = arith.addi %[[VAL_8]], %[[VAL_10]] : index
// CHECK-NEXT:    %[[VAL_12:.*]] = arith.maxsi %[[VAL_7]], %[[VAL_11]] : index
// CHECK-NEXT:    %[[VAL_13:.*]] = sycl.local_id : !sycl_id_1_1
// CHECK-NEXT:    %[[VAL_14:.*]] = memref.alloca() : memref<1x!sycl_id_1_1>
// CHECK-NEXT:    %[[VAL_15:.*]] = arith.constant 0 : index
// CHECK-NEXT:    memref.store %[[VAL_13]], %[[VAL_14]]{{\[}}%[[VAL_15]]] : memref<1x!sycl_id_1_1>
// CHECK-NEXT:    %[[VAL_16:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[VAL_17:.*]] = sycl.id.get %[[VAL_14]]{{\[}}%[[VAL_16]]] : (memref<1x!sycl_id_1_1>, i32) -> index
// CHECK-NEXT:    %[[VAL_18:.*]] = memref.alloca() : memref<1x!sycl_id_1_>
// CHECK-NEXT:    %[[VAL_19:.*]] = memref.cast %[[VAL_18]] : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
// CHECK-NEXT:    %[[VAL_20:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[VAL_21:.*]] = sycl.item.get_id(%[[VAL_1]], %[[VAL_20]]) : (memref<?x!sycl_item_1_>, i32) -> i64
// CHECK-NEXT:    %[[VAL_22:.*]] = arith.constant 32000 : index
// CHECK-NEXT:    %[[VAL_23:.*]] = arith.cmpi ule, %[[VAL_12]], %[[VAL_22]] : index
// CHECK-NEXT:    scf.if %[[VAL_23]] {
// CHECK-NEXT:      %[[VAL_24:.*]] = memref.get_global @WGLocalMem : memref<32000xi8, #sycl.access.address_space<local>>
// CHECK-NEXT:      affine.for %[[VAL_25:.*]] = 0 to [[MAP1]](){{\[}}%[[VAL_6]]] {
// CHECK-NEXT:        %[[VAL_26:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_27:.*]] = memref.view %[[VAL_24]]{{\[}}%[[VAL_26]]]{{\[}}%[[VAL_6]]] : memref<32000xi8, #sycl.access.address_space<local>> to memref<?xf32, #sycl.access.address_space<local>>
// CHECK-NEXT:        %[[VAL_28:.*]] = affine.apply [[MAP2]](%[[VAL_25]]){{\[}}%[[VAL_6]]]
// CHECK-NEXT:        %[[VAL_29:.*]] = arith.addi %[[VAL_17]], %[[VAL_28]] : index
// CHECK-NEXT:        %[[VAL_30:.*]] = arith.index_cast %[[VAL_29]] : index to i64
// CHECK-NEXT:        %[[VAL_31:.*]] = sycl.id.constructor(%[[VAL_29]]) : (index) -> memref<1x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_32:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_33:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_31]]] : (memref<?x!sycl_accessor_1_f32_r_dev>, memref<1x!sycl_id_1_>) -> memref<?xf32, 1>
// CHECK-NEXT:        %[[VAL_34:.*]] = memref.load %[[VAL_33]]{{\[}}%[[VAL_32]]] : memref<?xf32, 1>
// CHECK-NEXT:        memref.store %[[VAL_34]], %[[VAL_27]]{{\[}}%[[VAL_17]]] : memref<?xf32, #sycl.access.address_space<local>>
// CHECK-NEXT:        spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>

// COM: Calculate loop bounds:
// CHECK-NEXT:        %[[VAL_35:.*]] = arith.muli %[[VAL_25]], %[[VAL_6]] : index
// CHECK-NEXT:        %[[VAL_36:.*]] = arith.muli %[[VAL_25]], %[[VAL_6]] : index
// CHECK-NEXT:        %[[VAL_37:.*]] = arith.addi %[[VAL_36]], %[[VAL_6]] : index
// CHECK-NEXT:        %[[VAL_38:.*]] = arith.constant 256 : index
// CHECK-NEXT:        %[[VAL_39:.*]] = arith.cmpi slt, %[[VAL_37]], %[[VAL_38]] : index
// CHECK-NEXT:        %[[VAL_40:.*]] = arith.select %[[VAL_39]], %[[VAL_37]], %[[VAL_38]] : index
// CHECK-NEXT:        %[[VAL_41:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_42:.*]] = arith.subi %[[VAL_40]], %[[VAL_35]] : index
// CHECK-NEXT:        %[[VAL_43:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_44:.*]] = arith.subi %[[VAL_41]], %[[VAL_43]] : index
// CHECK-NEXT:        %[[VAL_45:.*]] = arith.addi %[[VAL_42]], %[[VAL_44]] : index
// CHECK-NEXT:        %[[VAL_46:.*]] = arith.divui %[[VAL_45]], %[[VAL_41]] : index
// CHECK-NEXT:        %[[VAL_47:.*]] = arith.constant 2 : index
// CHECK-NEXT:        %[[VAL_48:.*]] = arith.remsi %[[VAL_46]], %[[VAL_47]] : index
// CHECK-NEXT:        %[[VAL_49:.*]] = arith.subi %[[VAL_46]], %[[VAL_48]] : index
// CHECK-NEXT:        %[[VAL_50:.*]] = arith.muli %[[VAL_49]], %[[VAL_41]] : index
// CHECK-NEXT:        %[[VAL_51:.*]] = arith.addi %[[VAL_35]], %[[VAL_50]] : index
// CHECK-NEXT:        %[[VAL_52:.*]] = arith.muli %[[VAL_41]], %[[VAL_47]] : index

// COM: Tiled loop unrolled by 2:
// CHECK-NEXT:        scf.for %[[VAL_53:.*]] = %[[VAL_35]] to %[[VAL_51]] step %[[VAL_52]] {
// CHECK-NEXT:          %[[VAL_54:.*]] = arith.subi %[[VAL_53]], %[[VAL_28]] : index
// CHECK-NEXT:          %[[VAL_55:.*]] = arith.index_cast %[[VAL_53]] : index to i64
// CHECK-NEXT:          %[[VAL_56:.*]] = arith.index_cast %[[VAL_54]] : index to i64
// CHECK-NEXT:          sycl.constructor @id(%[[VAL_19]], %[[VAL_55]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_1_>, i64)
// CHECK-NEXT:          %[[VAL_57:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_54]]] : memref<?xf32, #sycl.access.address_space<local>>
// CHECK-NEXT:          %[[VAL_58:.*]] = arith.constant 1 : index
// CHECK-NEXT:          %[[VAL_59:.*]] = arith.muli %[[VAL_41]], %[[VAL_58]] : index
// CHECK-NEXT:          %[[VAL_60:.*]] = arith.addi %[[VAL_53]], %[[VAL_59]] : index
// CHECK-NEXT:          %[[VAL_61:.*]] = arith.subi %[[VAL_60]], %[[VAL_28]] : index
// CHECK-NEXT:          %[[VAL_62:.*]] = arith.index_cast %[[VAL_60]] : index to i64
// CHECK-NEXT:          %[[VAL_63:.*]] = arith.index_cast %[[VAL_61]] : index to i64
// CHECK-NEXT:          sycl.constructor @id(%[[VAL_19]], %[[VAL_62]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_1_>, i64)
// CHECK-NEXT:          %[[VAL_64:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_61]]] : memref<?xf32, #sycl.access.address_space<local>>
// CHECK-NEXT:        }

// COM: Remainder loop after loop unrolling:
// CHECK-NEXT:        scf.for %[[VAL_65:.*]] = %[[VAL_51]] to %[[VAL_40]] step %[[VAL_41]] {
// CHECK-NEXT:          %[[VAL_66:.*]] = arith.subi %[[VAL_65]], %[[VAL_28]] : index
// CHECK-NEXT:          %[[VAL_67:.*]] = arith.index_cast %[[VAL_65]] : index to i64
// CHECK-NEXT:          %[[VAL_68:.*]] = arith.index_cast %[[VAL_66]] : index to i64
// CHECK-NEXT:          sycl.constructor @id(%[[VAL_19]], %[[VAL_67]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_1_>, i64)
// CHECK-NEXT:          %[[VAL_69:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_66]]] : memref<?xf32, #sycl.access.address_space<local>>
// CHECK-NEXT:        }

// CHECK-NEXT:        spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:      }
// CHECK-NEXT:    } else {
// CHECK-NEXT:      affine.for %[[VAL_70:.*]] = 0 to 256 {
// CHECK-NEXT:        %[[VAL_71:.*]] = arith.index_cast %[[VAL_70]] : index to i64
// CHECK-NEXT:        sycl.constructor @id(%[[VAL_19]], %[[VAL_71]]) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_1_>, i64)
// CHECK-NEXT:        %[[VAL_72:.*]] = sycl.accessor.subscript %[[VAL_0]]{{\[}}%[[VAL_19]]] : (memref<?x!sycl_accessor_1_f32_r_dev>, memref<?x!sycl_id_1_>) -> memref<?xf32>
// CHECK-NEXT:        %[[VAL_73:.*]] = affine.load %[[VAL_72]][0] : memref<?xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
gpu.module @device_func {
func.func private @affine(%arg0: memref<?x!sycl_accessor_1_f32_r_dev>, %arg1: memref<?x!sycl_item_1_>) {
  %alloca = memref.alloca() : memref<1x!sycl_id_1_>
  %id = memref.cast %alloca : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
  %c0_i32 = arith.constant 0 : i32
  %tx = sycl.item.get_id(%arg1, %c0_i32) : (memref<?x!sycl_item_1_>, i32) -> i64

  affine.for %ii = 0 to 256 {
    %i = arith.index_cast %ii : index to i64
    sycl.constructor @id(%id, %i) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_1_>, i64)
    %subscr1 = sycl.accessor.subscript %arg0[%id] : (memref<?x!sycl_accessor_1_f32_r_dev>, memref<?x!sycl_id_1_>) -> memref<?xf32>
    %load1 = affine.load %subscr1[0] : memref<?xf32>
  }
  return
}
gpu.func @kernel(%arg0: memref<?x!sycl_accessor_1_f32_r_dev>, %arg1: memref<?x!sycl_item_1_>) kernel {
  func.call @affine(%arg0, %arg1) : (memref<?x!sycl_accessor_1_f32_r_dev>, memref<?x!sycl_item_1_>) -> ()
  gpu.return
}
}
