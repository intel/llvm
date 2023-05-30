// RUN: polygeist-opt --loop-internalization --split-input-file -allow-unregistered-dialect %s | FileCheck %s --check-prefixes=CHECK,SIZE1
// RUN: polygeist-opt --loop-internalization --loop-internalization-tile-sizes=2 --split-input-file -allow-unregistered-dialect %s | FileCheck %s --check-prefixes=CHECK,SIZE2

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_accessor_1_f32_r_gb = !sycl.accessor<[1, f32, read, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xf32, 1>)>)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_, !sycl_group_1_)>
// CHECK-DAG:   [[MAP1:#map.*]] = affine_map<()[s0] -> (256 ceildiv s0)>
// CHECK-DAG:   [[MAP2:#map.*]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG:   [[MAP3:#map.*]] = affine_map<(d0)[s0] -> (d0 * s0 + s0, 256)>
// CHECK-LABEL: func.func private @affine_1d(%arg0: memref<?x!sycl_accessor_1_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_1_>) {
// SIZE1-NEXT:    [[TILESIZE:%.*]] = arith.constant 1 : index
// SIZE2-NEXT:    [[TILESIZE:%.*]] = arith.constant 2 : index
// CHECK-NEXT:    affine.for [[IV1:%.*]] = 0 to [[MAP1]]()[[[TILESIZE]]] {
// CHECK-NEXT:      spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:      affine.for [[IV2:%.*]] = [[MAP2]]([[IV1]])[[[TILESIZE]]] to min [[MAP3]]([[IV1]])[[[TILESIZE]]] {
// CHECK-NEXT:        "test.foo"([[IV2]]) : (index) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
gpu.module @device_func {
func.func private @affine_1d(%arg0: memref<?x!sycl_accessor_1_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_1_>) {
  affine.for %i = 0 to 256 {
    "test.foo"(%i) : (index) -> ()
  }
  return
}
gpu.func @kernel(%arg0: memref<?x!sycl_accessor_1_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_1_>) kernel {
  func.call @affine_1d(%arg0, %arg1) : (memref<?x!sycl_accessor_1_f32_r_gb>, memref<?x!sycl_nd_item_1_>) -> ()
  gpu.return
}
}

// -----

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_accessor_1_f32_r_gb = !sycl.accessor<[1, f32, read, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xf32, 1>)>)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_, !sycl_group_1_)>
// CHECK-DAG:   [[MAP1:#map.*]] = affine_map<()[s0] -> (256 ceildiv s0)>
// CHECK-DAG:   [[MAP2:#map.*]] = affine_map<()[s0] -> (511 ceildiv s0 + 1)>
// CHECK-DAG:   [[MAP3:#map.*]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG:   [[MAP4:#map.*]] = affine_map<(d0)[s0] -> (d0 * s0 + s0, 256)>
// CHECK-DAG:   [[MAP5:#map.*]] = affine_map<(d0)[s0] -> ((d0 - 1) * s0 + 1)>
// CHECK-DAG:   [[MAP6:#map.*]] = affine_map<(d0)[s0] -> ((d0 - 1) * s0 + s0 + 1, 512)>
// CHECK-LABEL: func.func private @affine_2d(%arg0: memref<?x!sycl_accessor_1_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_1_>) {
// SIZE1-NEXT:    [[TILESIZE:%.*]] = arith.constant 1 : index
// SIZE2-NEXT:    [[TILESIZE:%.*]] = arith.constant 2 : index
// SIZE2-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    affine.for [[IV1:%.*]] = 0 to [[MAP1]]()[[[TILESIZE]]] {
// CHECK-NEXT:      affine.for [[IV2:%.*]] = 1 to [[MAP2]]()[%c1] {
// CHECK-NEXT:        spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:        affine.for [[IV3:%.*]] = [[MAP3]]([[IV1]])[[[TILESIZE]]] to min [[MAP4]]([[IV1]])[[[TILESIZE]]] {
// CHECK-NEXT:          affine.for [[IV4:%.*]] = [[MAP5]]([[IV2]])[%c1] to min [[MAP6]]([[IV2]])[%c1] {
// CHECK-NEXT:            "test.foo"([[IV3]], [[IV4]]) : (index, index) -> ()
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
gpu.module @device_func {
func.func private @affine_2d(%arg0: memref<?x!sycl_accessor_1_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_1_>) {
  affine.for %i = 0 to 256 {
    affine.for %j = 1 to 512 {
      "test.foo"(%i, %j) : (index, index) -> ()
    }
  }
  return
}
gpu.func @kernel(%arg0: memref<?x!sycl_accessor_1_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_1_>) kernel {
  func.call @affine_2d(%arg0, %arg1) : (memref<?x!sycl_accessor_1_f32_r_gb>, memref<?x!sycl_nd_item_1_>) -> ()
  gpu.return
}
}

// -----

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_accessor_1_f32_r_gb = !sycl.accessor<[1, f32, read, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xf32, 1>)>)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_, !sycl_group_1_)>
// CHECK-LABEL: func.func private @scf_1d(%arg0: memref<?x!sycl_accessor_1_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_1_>) {
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c256 = arith.constant 256 : index
// SIZE1-DAG:     [[TILESIZE:%.*]] = arith.constant 1 : index
// SIZE2-DAG:     [[TILESIZE:%.*]] = arith.constant 2 : index
// CHECK-NEXT:    %0 = arith.muli %c1, [[TILESIZE]] : index
// CHECK-NEXT:    scf.for [[IV1:%.*]] = %c0 to %c256 step %0 {
// CHECK-NEXT:      spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:      [[VAL_1:%.*]] = arith.addi [[IV1]], %0 : index
// CHECK-NEXT:      [[VAL_2:%.*]] = arith.cmpi slt, %c256, [[VAL_1]] : index
// CHECK-NEXT:      [[VAL_3:%.*]] = arith.select [[VAL_2]], %c256, [[VAL_1]] : index
// CHECK-NEXT:      scf.for [[IV2:%.*]] = [[IV1]] to [[VAL_3]] step %c1 {
// CHECK-NEXT:        "test.foo"([[IV2]]) : (index) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
gpu.module @device_func {
func.func private @scf_1d(%arg0: memref<?x!sycl_accessor_1_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_1_>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  scf.for %i = %c0 to %c256 step %c1 {
    "test.foo"(%i) : (index) -> ()
  }
  return
}
gpu.func @kernel(%arg0: memref<?x!sycl_accessor_1_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_1_>) kernel {
  func.call @scf_1d(%arg0, %arg1) : (memref<?x!sycl_accessor_1_f32_r_gb>, memref<?x!sycl_nd_item_1_>) -> ()
  gpu.return
}
}

// -----

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_group_1_ = !sycl.group<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_accessor_1_f32_r_gb = !sycl.accessor<[1, f32, read, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xf32, 1>)>)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>
!sycl_nd_item_1_ = !sycl.nd_item<[1], (!sycl_item_1_, !sycl_item_1_, !sycl_group_1_)>
// CHECK-LABEL: func.func private @scf_2d(%arg0: memref<?x!sycl_accessor_1_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_1_>) {
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c256 = arith.constant 256 : index
// CHECK-DAG:     %c512 = arith.constant 512 : index
// SIZE1-DAG:     [[TILESIZE:%.*]] = arith.constant 1 : index
// SIZE2-DAG:     [[TILESIZE:%.*]] = arith.constant 2 : index
// SIZE2-DAG:     %c1_0 = arith.constant 1 : index
// CHECK-NEXT:    %0 = arith.muli %c1, [[TILESIZE]] : index
// CHECK-NEXT:    scf.for [[IV1:.*]] = %c0 to %c256 step %0 {
// CHECK-NEXT:      %1 = arith.muli %c1, %c1_0 : index
// CHECK-NEXT:      scf.for [[IV2:.*]] = %c1 to %c512 step %1 {
// CHECK-NEXT:        spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:        [[VAL_2:%.*]] = arith.addi [[IV1]], %0 : index
// CHECK-NEXT:        [[VAL_3:%.*]] = arith.cmpi slt, %c256, [[VAL_2]] : index
// CHECK-NEXT:        [[VAL_4:%.*]] = arith.select [[VAL_3]], %c256, [[VAL_2]] : index
// CHECK-NEXT:        scf.for [[IV3:.*]] = [[IV1]] to [[VAL_4]] step %c1 {
// CHECK-NEXT:          [[VAL_5:%.*]] = arith.addi [[IV2]], %1 : index
// CHECK-NEXT:          [[VAL_6:%.*]] = arith.cmpi slt, %c512, [[VAL_5]] : index
// CHECK-NEXT:          [[VAL_7:%.*]] = arith.select [[VAL_6]], %c512, [[VAL_5]] : index
// CHECK-NEXT:          scf.for [[IV4:.*]] = [[IV2]] to [[VAL_7]] step %c1 {
// CHECK-NEXT:            "test.foo"([[IV3]], [[IV4]]) : (index, index) -> ()
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
gpu.module @device_func {
func.func private @scf_2d(%arg0: memref<?x!sycl_accessor_1_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_1_>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  scf.for %i = %c0 to %c256 step %c1 {
    scf.for %j = %c1 to %c512 step %c1 {
      "test.foo"(%i, %j) : (index, index) -> ()
    }
  }
  return
}
gpu.func @kernel(%arg0: memref<?x!sycl_accessor_1_f32_r_gb>, %arg1: memref<?x!sycl_nd_item_1_>) kernel {
  func.call @scf_2d(%arg0, %arg1) : (memref<?x!sycl_accessor_1_f32_r_gb>, memref<?x!sycl_nd_item_1_>) -> ()
  gpu.return
}
}
