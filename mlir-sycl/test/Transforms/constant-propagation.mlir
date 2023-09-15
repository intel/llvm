// RUN: sycl-mlir-opt %s -split-input-file -sycl-constant-propagation | FileCheck %s

// COM: Check we do not break if no launch is found.

gpu.module @kernels {
// CHECK-LABEL: gpu.func @k0
  gpu.func @k0(%ptr: memref<?xi64, 1>,
               %c: i64) kernel {
    affine.store %c, %ptr[0] : memref<?xi64, 1>
    gpu.return
  }
}

// -----

// COM: Check we do not allow more than one launches

gpu.module @kernels {
// CHECK-LABEL:     gpu.func @k0(
// CHECK-SAME:                   %[[VAL_0:.*]]: memref<?xi64, 1>, %[[VAL_1:.*]]: i64) kernel {
// CHECK-NEXT:        affine.store %[[VAL_1]], %[[VAL_0]][0] : memref<?xi64, 1>
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }
  gpu.func @k0(%ptr: memref<?xi64, 1>,
               %c: i64) kernel {
    affine.store %c, %ptr[0] : memref<?xi64, 1>
    gpu.return
  }
}

llvm.func internal @foo.1(%handler: !llvm.ptr, %ptr: !llvm.ptr) {
  %c = llvm.mlir.constant(0 : i64) : i64
  sycl.host.schedule_kernel %handler -> @kernels::@k0(%ptr, %c)
      : (!llvm.ptr, !llvm.ptr, i64) -> ()
  llvm.return
}

llvm.func internal @foo.2(%handler: !llvm.ptr, %ptr: !llvm.ptr) {
  %c = llvm.mlir.constant(0 : i64) : i64
  sycl.host.schedule_kernel %handler -> @kernels::@k0(%ptr, %c)
      : (!llvm.ptr, !llvm.ptr, i64) -> ()
  llvm.return
}

// -----

// COM: Check we allow other users

gpu.module @kernels {
// CHECK-LABEL:     gpu.func @k0(
// CHECK-SAME:                   %[[VAL_0:.*]]: memref<?xi64, 1>, %[[VAL_1:.*]]: i64) kernel {
// CHECK-NEXT:        %[[C:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:        affine.store %[[C]], %[[VAL_0]][0] : memref<?xi64, 1>
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }
  gpu.func @k0(%ptr: memref<?xi64, 1>,
               %c: i64) kernel {
    affine.store %c, %ptr[0] : memref<?xi64, 1>
    gpu.return
  }
}

llvm.func internal @foo.1(%handler: !llvm.ptr, %ptr: !llvm.ptr) {
  %c = llvm.mlir.constant(0 : i64) : i64
  sycl.host.handler.set_kernel %handler -> @kernels::@k0 : !llvm.ptr
  sycl.host.schedule_kernel %handler -> @kernels::@k0(%ptr, %c)
      : (!llvm.ptr, !llvm.ptr, i64) -> ()
  llvm.return
}

// -----

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
!sycl_accessor_1_i64_w_gb = !sycl.accessor<[1, i64, write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>, !llvm.struct<(ptr<i32, 1>)>)>

// COM: Check we can detect kernel argument %c is constant

gpu.module @kernels {
  func.func private @init(%acc: memref<1x!sycl_accessor_1_i64_w_gb>,
                          %ptr: memref<?xi64, 1>,
                          %accRange: memref<?x!sycl_range_1_>,
                          %memRange: memref<?x!sycl_range_1_>,
                          %offset: memref<?x!sycl_id_1_>)

// CHECK-LABEL:     gpu.func @k0(
// CHECK-SAME:                   %[[VAL_0:.*]]: memref<?xi64, 1>, %[[VAL_1:.*]]: memref<?x!sycl_range_1_>, %[[VAL_2:.*]]: memref<?x!sycl_range_1_>, %[[VAL_3:.*]]: memref<?x!sycl_id_1_>, %[[VAL_4:.*]]: i64) kernel {
// CHECK-NEXT:        %[[VAL_5:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:        %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_7:.*]] = arith.constant 0 : i64
// CHECK-NEXT:        %[[VAL_8:.*]] = memref.alloca() : memref<1x!sycl_accessor_1_i64_w_gb>
// CHECK-NEXT:        func.call @init(%[[VAL_8]], %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) : (memref<1x!sycl_accessor_1_i64_w_gb>, memref<?xi64, 1>, memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>, memref<?x!sycl_id_1_>) -> ()
// CHECK-NEXT:        %[[VAL_9:.*]] = sycl.accessor.subscript %[[VAL_8]]{{\[}}%[[VAL_7]]] : (memref<1x!sycl_accessor_1_i64_w_gb>, i64) -> memref<?xi64>
// CHECK-NEXT:        memref.store %[[VAL_5]], %[[VAL_9]]{{\[}}%[[VAL_6]]] : memref<?xi64>
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }
  gpu.func @k0(%ptr: memref<?xi64, 1>,
               %accRange: memref<?x!sycl_range_1_>,
               %memRange: memref<?x!sycl_range_1_>,
               %offset: memref<?x!sycl_id_1_>,
               %c: i64) kernel {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %acc = memref.alloca() : memref<1x!sycl_accessor_1_i64_w_gb>
    func.call @init(%acc, %ptr, %accRange, %memRange, %offset)
        : (memref<1x!sycl_accessor_1_i64_w_gb>, memref<?xi64, 1>,
           memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>,
           memref<?x!sycl_id_1_>) -> ()
    %res = sycl.accessor.subscript %acc[%c0_i64]
        : (memref<1x!sycl_accessor_1_i64_w_gb>, i64) -> memref<?xi64>
    memref.store %c, %res[%c0] : memref<?xi64>
    gpu.return
  }
}

llvm.func internal @foo(%handler: !llvm.ptr, %acc: !llvm.ptr) {
  %c = llvm.mlir.constant(0 : i64) : i64
  sycl.host.schedule_kernel %handler -> @kernels::@k0(%acc: !sycl_accessor_1_i64_w_gb, %c)
      : (!llvm.ptr, !llvm.ptr, i64) -> ()
  llvm.return
}

// -----

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
!sycl_array_2_ = !sycl.array<[2], (memref<2xi64>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl_array_2_)>
!sycl_range_2_ = !sycl.range<[2], (!sycl_array_2_)>
!sycl_array_3_ = !sycl.array<[3], (memref<3xi64>)>
!sycl_id_3_ = !sycl.id<[3], (!sycl_array_3_)>
!sycl_range_3_ = !sycl.range<[3], (!sycl_array_3_)>
!sycl_nd_range_3_ =
    !sycl.nd_range<[3], (!sycl_range_3_, !sycl_range_3_, !sycl_id_3_)>

gpu.module @kernels {
// COM: `@__init.*` functions will see the actual builtin replacement happening,
// COM: e.g., in `@__init.1_k0_impl.1`, we see `sycl.global_offset` being
// COM: replaced by a constant `sycl.id` (`%[[VAL_6]]` below).

// CHECK-LABEL:     func.func @__init.1_k0_impl.1(
// CHECK-SAME:                                    %[[VAL_0:.*]]: memref<1x!sycl_id_1_>, %[[VAL_1:.*]]: memref<1x!sycl_range_1_>, %[[VAL_2:.*]]: memref<1x!sycl_range_1_>, %[[VAL_3:.*]]: memref<1x!sycl_range_1_>) {
// CHECK-NEXT:        %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_5:.*]] = sycl.id.constructor() : () -> memref<1x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_6:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_4]]] : memref<1x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_9:.*]] = sycl.num_work_items : !sycl_range_1_
// CHECK-NEXT:        %[[VAL_10:.*]] = sycl.work_group_size : !sycl_range_1_
// CHECK-NEXT:        %[[VAL_11:.*]] = sycl.num_work_groups : !sycl_range_1_
// CHECK-NEXT:        memref.store %[[VAL_6]], %[[VAL_0]]{{\[}}%[[VAL_7]]] : memref<1x!sycl_id_1_>
// CHECK-NEXT:        memref.store %[[VAL_9]], %[[VAL_1]]{{\[}}%[[VAL_7]]] : memref<1x!sycl_range_1_>
// CHECK-NEXT:        memref.store %[[VAL_10]], %[[VAL_2]]{{\[}}%[[VAL_7]]] : memref<1x!sycl_range_1_>
// CHECK-NEXT:        memref.store %[[VAL_11]], %[[VAL_3]]{{\[}}%[[VAL_7]]] : memref<1x!sycl_range_1_>
// CHECK-NEXT:        return
// CHECK-NEXT:      }

// CHECK-LABEL:     func.func @__init.1_k1_impl.1(
// CHECK-SAME:                                    %[[VAL_12:.*]]: memref<1x!sycl_id_1_>, %[[VAL_13:.*]]: memref<1x!sycl_range_1_>, %[[VAL_14:.*]]: memref<1x!sycl_range_1_>, %[[VAL_15:.*]]: memref<1x!sycl_range_1_>) {
// CHECK-NEXT:        %[[VAL_16:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_17:.*]] = arith.constant 512 : index
// CHECK-NEXT:        %[[VAL_18:.*]] = sycl.range.constructor(%[[VAL_17]]) : (index) -> memref<1x!sycl_range_1_>
// CHECK-NEXT:        %[[VAL_19:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_16]]] : memref<1x!sycl_range_1_>
// CHECK-NEXT:        %[[VAL_20:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_21:.*]] = sycl.global_offset : !sycl_id_1_
// CHECK-NEXT:        %[[VAL_23:.*]] = sycl.work_group_size : !sycl_range_1_
// CHECK-NEXT:        %[[VAL_24:.*]] = sycl.num_work_groups : !sycl_range_1_
// CHECK-NEXT:        memref.store %[[VAL_21]], %[[VAL_12]]{{\[}}%[[VAL_20]]] : memref<1x!sycl_id_1_>
// CHECK-NEXT:        memref.store %[[VAL_19]], %[[VAL_13]]{{\[}}%[[VAL_20]]] : memref<1x!sycl_range_1_>
// CHECK-NEXT:        memref.store %[[VAL_23]], %[[VAL_14]]{{\[}}%[[VAL_20]]] : memref<1x!sycl_range_1_>
// CHECK-NEXT:        memref.store %[[VAL_24]], %[[VAL_15]]{{\[}}%[[VAL_20]]] : memref<1x!sycl_range_1_>
// CHECK-NEXT:        return
// CHECK-NEXT:      }
  func.func @init.1(%offset: memref<1x!sycl_id_1_>,
                      %global_size: memref<1x!sycl_range_1_>,
                      %local_size: memref<1x!sycl_range_1_>,
                      %num_work_groups: memref<1x!sycl_range_1_>) {
    %c0 = arith.constant 0 : index
    %offset.val = sycl.global_offset : !sycl_id_1_
    %global_size.val = sycl.num_work_items : !sycl_range_1_
    %local_size.val = sycl.work_group_size : !sycl_range_1_
    %num_work_groups.val = sycl.num_work_groups : !sycl_range_1_
    memref.store %offset.val, %offset[%c0] : memref<1x!sycl_id_1_>
    memref.store %global_size.val, %global_size[%c0] : memref<1x!sycl_range_1_>
    memref.store %local_size.val, %local_size[%c0] : memref<1x!sycl_range_1_>
    memref.store %num_work_groups.val, %num_work_groups[%c0]
        : memref<1x!sycl_range_1_>
    func.return
  }

// CHECK-LABEL:     func.func @__init.2_k2_impl.2(
// CHECK-SAME:                                    %[[VAL_34:.*]]: memref<1x!sycl_id_2_>, %[[VAL_35:.*]]: memref<1x!sycl_range_2_>, %[[VAL_36:.*]]: memref<1x!sycl_range_2_>, %[[VAL_37:.*]]: memref<1x!sycl_range_2_>) {
// CHECK-NEXT:        %[[VAL_38:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_39:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_40:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_41:.*]] = sycl.id.constructor(%[[VAL_39]], %[[VAL_40]]) : (index, index) -> memref<1x!sycl_id_2_>
// CHECK-NEXT:        %[[VAL_42:.*]] = memref.load %[[VAL_41]]{{\[}}%[[VAL_38]]] : memref<1x!sycl_id_2_>
// CHECK-NEXT:        %[[VAL_43:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_44:.*]] = arith.constant 512 : index
// CHECK-NEXT:        %[[VAL_45:.*]] = arith.constant 512 : index
// CHECK-NEXT:        %[[VAL_46:.*]] = sycl.range.constructor(%[[VAL_44]], %[[VAL_45]]) : (index, index) -> memref<1x!sycl_range_2_>
// CHECK-NEXT:        %[[VAL_47:.*]] = memref.load %[[VAL_46]]{{\[}}%[[VAL_43]]] : memref<1x!sycl_range_2_>
// CHECK-NEXT:        %[[VAL_48:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_51:.*]] = sycl.work_group_size : !sycl_range_2_
// CHECK-NEXT:        %[[VAL_52:.*]] = sycl.num_work_groups : !sycl_range_2_
// CHECK-NEXT:        memref.store %[[VAL_42]], %[[VAL_34]]{{\[}}%[[VAL_48]]] : memref<1x!sycl_id_2_>
// CHECK-NEXT:        memref.store %[[VAL_47]], %[[VAL_35]]{{\[}}%[[VAL_48]]] : memref<1x!sycl_range_2_>
// CHECK-NEXT:        memref.store %[[VAL_51]], %[[VAL_36]]{{\[}}%[[VAL_48]]] : memref<1x!sycl_range_2_>
// CHECK-NEXT:        memref.store %[[VAL_52]], %[[VAL_37]]{{\[}}%[[VAL_48]]] : memref<1x!sycl_range_2_>
// CHECK-NEXT:        return
// CHECK-NEXT:      }
  func.func @init.2(%offset: memref<1x!sycl_id_2_>,
                      %global_size: memref<1x!sycl_range_2_>,
                      %local_size: memref<1x!sycl_range_2_>,
                      %num_work_groups: memref<1x!sycl_range_2_>) {
    %c0 = arith.constant 0 : index
    %offset.val = sycl.global_offset : !sycl_id_2_
    %global_size.val = sycl.num_work_items : !sycl_range_2_
    %local_size.val = sycl.work_group_size : !sycl_range_2_
    %num_work_groups.val = sycl.num_work_groups : !sycl_range_2_
    memref.store %offset.val, %offset[%c0] : memref<1x!sycl_id_2_>
    memref.store %global_size.val, %global_size[%c0] : memref<1x!sycl_range_2_>
    memref.store %local_size.val, %local_size[%c0] : memref<1x!sycl_range_2_>
    memref.store %num_work_groups.val, %num_work_groups[%c0]
        : memref<1x!sycl_range_2_>
    func.return
  }

// CHECK-LABEL:     func.func @__init.3_k3_impl.3(
// CHECK-SAME:                                    %[[VAL_62:.*]]: memref<1x!sycl_id_3_>, %[[VAL_63:.*]]: memref<1x!sycl_range_3_>, %[[VAL_64:.*]]: memref<1x!sycl_range_3_>, %[[VAL_65:.*]]: memref<1x!sycl_range_3_>) {
// CHECK-NEXT:        %[[VAL_66:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_67:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_68:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_69:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_70:.*]] = sycl.id.constructor(%[[VAL_67]], %[[VAL_68]], %[[VAL_69]]) : (index, index, index) -> memref<1x!sycl_id_3_>
// CHECK-NEXT:        %[[VAL_71:.*]] = memref.load %[[VAL_70]]{{\[}}%[[VAL_66]]] : memref<1x!sycl_id_3_>
// CHECK-NEXT:        %[[VAL_72:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_74:.*]] = sycl.num_work_items : !sycl_range_3_
// CHECK-NEXT:        %[[VAL_75:.*]] = sycl.work_group_size : !sycl_range_3_
// CHECK-NEXT:        %[[VAL_76:.*]] = sycl.num_work_groups : !sycl_range_3_
// CHECK-NEXT:        memref.store %[[VAL_71]], %[[VAL_62]]{{\[}}%[[VAL_72]]] : memref<1x!sycl_id_3_>
// CHECK-NEXT:        memref.store %[[VAL_74]], %[[VAL_63]]{{\[}}%[[VAL_72]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        memref.store %[[VAL_75]], %[[VAL_64]]{{\[}}%[[VAL_72]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        memref.store %[[VAL_76]], %[[VAL_65]]{{\[}}%[[VAL_72]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        return
// CHECK-NEXT:      }

// CHECK-LABEL:     func.func @__init.3_k4_impl.3(
// CHECK-SAME:                                    %[[VAL_77:.*]]: memref<1x!sycl_id_3_>, %[[VAL_78:.*]]: memref<1x!sycl_range_3_>, %[[VAL_79:.*]]: memref<1x!sycl_range_3_>, %[[VAL_80:.*]]: memref<1x!sycl_range_3_>) {
// CHECK-NEXT:        %[[VAL_81:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_82:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_83:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_84:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_85:.*]] = sycl.id.constructor(%[[VAL_82]], %[[VAL_83]], %[[VAL_84]]) : (index, index, index) -> memref<1x!sycl_id_3_>
// CHECK-NEXT:        %[[VAL_86:.*]] = memref.load %[[VAL_85]]{{\[}}%[[VAL_81]]] : memref<1x!sycl_id_3_>
// CHECK-NEXT:        %[[VAL_87:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_89:.*]] = sycl.num_work_items : !sycl_range_3_
// CHECK-NEXT:        %[[VAL_90:.*]] = sycl.work_group_size : !sycl_range_3_
// CHECK-NEXT:        %[[VAL_91:.*]] = sycl.num_work_groups : !sycl_range_3_
// CHECK-NEXT:        memref.store %[[VAL_86]], %[[VAL_77]]{{\[}}%[[VAL_87]]] : memref<1x!sycl_id_3_>
// CHECK-NEXT:        memref.store %[[VAL_89]], %[[VAL_78]]{{\[}}%[[VAL_87]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        memref.store %[[VAL_90]], %[[VAL_79]]{{\[}}%[[VAL_87]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        memref.store %[[VAL_91]], %[[VAL_80]]{{\[}}%[[VAL_87]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        return
// CHECK-NEXT:      }

// CHECK-LABEL:     func.func @__init.3_k5_impl.3(
// CHECK-SAME:                                    %[[VAL_92:.*]]: memref<1x!sycl_id_3_>, %[[VAL_93:.*]]: memref<1x!sycl_range_3_>, %[[VAL_94:.*]]: memref<1x!sycl_range_3_>, %[[VAL_95:.*]]: memref<1x!sycl_range_3_>) {
// CHECK-NEXT:        %[[VAL_96:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_97:.*]] = arith.constant 512 : index
// CHECK-NEXT:        %[[VAL_98:.*]] = arith.constant 512 : index
// CHECK-NEXT:        %[[VAL_99:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_100:.*]] = sycl.range.constructor(%[[VAL_97]], %[[VAL_98]], %[[VAL_99]]) : (index, index, index) -> memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_101:.*]] = memref.load %[[VAL_100]]{{\[}}%[[VAL_96]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_102:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_103:.*]] = sycl.global_offset : !sycl_id_3_
// CHECK-NEXT:        %[[VAL_105:.*]] = sycl.work_group_size : !sycl_range_3_
// CHECK-NEXT:        %[[VAL_106:.*]] = sycl.num_work_groups : !sycl_range_3_
// CHECK-NEXT:        memref.store %[[VAL_103]], %[[VAL_92]]{{\[}}%[[VAL_102]]] : memref<1x!sycl_id_3_>
// CHECK-NEXT:        memref.store %[[VAL_101]], %[[VAL_93]]{{\[}}%[[VAL_102]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        memref.store %[[VAL_105]], %[[VAL_94]]{{\[}}%[[VAL_102]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        memref.store %[[VAL_106]], %[[VAL_95]]{{\[}}%[[VAL_102]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        return
// CHECK-NEXT:      }

// CHECK-LABEL:     func.func @__init.3_k6_impl.3(
// CHECK-SAME:                                    %[[VAL_107:.*]]: memref<1x!sycl_id_3_>, %[[VAL_108:.*]]: memref<1x!sycl_range_3_>, %[[VAL_109:.*]]: memref<1x!sycl_range_3_>, %[[VAL_110:.*]]: memref<1x!sycl_range_3_>) {
// CHECK-NEXT:        %[[VAL_111:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_112:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_113:.*]] = arith.constant 512 : index
// CHECK-NEXT:        %[[VAL_114:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_115:.*]] = sycl.range.constructor(%[[VAL_112]], %[[VAL_113]], %[[VAL_114]]) : (index, index, index) -> memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_116:.*]] = memref.load %[[VAL_115]]{{\[}}%[[VAL_111]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_117:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_118:.*]] = sycl.global_offset : !sycl_id_3_
// CHECK-NEXT:        %[[VAL_119:.*]] = sycl.num_work_items : !sycl_range_3_
// CHECK-NEXT:        %[[VAL_121:.*]] = sycl.num_work_groups : !sycl_range_3_
// CHECK-NEXT:        memref.store %[[VAL_118]], %[[VAL_107]]{{\[}}%[[VAL_117]]] : memref<1x!sycl_id_3_>
// CHECK-NEXT:        memref.store %[[VAL_119]], %[[VAL_108]]{{\[}}%[[VAL_117]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        memref.store %[[VAL_116]], %[[VAL_109]]{{\[}}%[[VAL_117]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        memref.store %[[VAL_121]], %[[VAL_110]]{{\[}}%[[VAL_117]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        return
// CHECK-NEXT:      }

// CHECK-LABEL:     func.func @__init.3_k7_impl.3(
// CHECK-SAME:                                    %[[VAL_122:.*]]: memref<1x!sycl_id_3_>, %[[VAL_123:.*]]: memref<1x!sycl_range_3_>, %[[VAL_124:.*]]: memref<1x!sycl_range_3_>, %[[VAL_125:.*]]: memref<1x!sycl_range_3_>) {
// CHECK-NEXT:        %[[VAL_126:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_127:.*]] = arith.constant 512 : index
// CHECK-NEXT:        %[[VAL_128:.*]] = arith.constant 512 : index
// CHECK-NEXT:        %[[VAL_129:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_130:.*]] = sycl.range.constructor(%[[VAL_127]], %[[VAL_128]], %[[VAL_129]]) : (index, index, index) -> memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_131:.*]] = memref.load %[[VAL_130]]{{\[}}%[[VAL_126]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_132:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_133:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_134:.*]] = arith.constant 512 : index
// CHECK-NEXT:        %[[VAL_135:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_136:.*]] = sycl.range.constructor(%[[VAL_133]], %[[VAL_134]], %[[VAL_135]]) : (index, index, index) -> memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_137:.*]] = memref.load %[[VAL_136]]{{\[}}%[[VAL_132]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_138:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_139:.*]] = arith.constant 512 : index
// CHECK-NEXT:        %[[VAL_140:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_141:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_142:.*]] = sycl.range.constructor(%[[VAL_139]], %[[VAL_140]], %[[VAL_141]]) : (index, index, index) -> memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_143:.*]] = memref.load %[[VAL_142]]{{\[}}%[[VAL_138]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_144:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_145:.*]] = sycl.global_offset : !sycl_id_3_
// CHECK-NEXT:        memref.store %[[VAL_145]], %[[VAL_122]]{{\[}}%[[VAL_144]]] : memref<1x!sycl_id_3_>
// CHECK-NEXT:        memref.store %[[VAL_131]], %[[VAL_123]]{{\[}}%[[VAL_144]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        memref.store %[[VAL_137]], %[[VAL_124]]{{\[}}%[[VAL_144]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        memref.store %[[VAL_143]], %[[VAL_125]]{{\[}}%[[VAL_144]]] : memref<1x!sycl_range_3_>
// CHECK-NEXT:        return
// CHECK-NEXT:      }
  func.func @init.3(%offset: memref<1x!sycl_id_3_>,
                      %global_size: memref<1x!sycl_range_3_>,
                      %local_size: memref<1x!sycl_range_3_>,
                      %num_work_groups: memref<1x!sycl_range_3_>) {
    %c0 = arith.constant 0 : index
    %offset.val = sycl.global_offset : !sycl_id_3_
    %global_size.val = sycl.num_work_items : !sycl_range_3_
    %local_size.val = sycl.work_group_size : !sycl_range_3_
    %num_work_groups.val = sycl.num_work_groups : !sycl_range_3_
    memref.store %offset.val, %offset[%c0] : memref<1x!sycl_id_3_>
    memref.store %global_size.val, %global_size[%c0] : memref<1x!sycl_range_3_>
    memref.store %local_size.val, %local_size[%c0] : memref<1x!sycl_range_3_>
    memref.store %num_work_groups.val, %num_work_groups[%c0]
        : memref<1x!sycl_range_3_>
    func.return
  }

// COM: `@__impl.*` and `k.*` functions will just suffer one change: the
// COM: `func.call` will now call the cloned function instead of the original
// COM: one, in order to get the actual constant propagation taking place in the
// COM: cloned function.

// CHECK-LABEL:     func.func @__impl.1_k0(
// CHECK-SAME:                             %[[VAL_158:.*]]: memref<?xindex>, %[[VAL_159:.*]]: memref<?xindex>, %[[VAL_160:.*]]: memref<?xindex>, %[[VAL_161:.*]]: memref<?xindex>) {
// CHECK-NEXT:        %[[VAL_162:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_163:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[VAL_164:.*]] = memref.alloca() : memref<1x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_165:.*]] = memref.alloca() : memref<1x!sycl_range_1_>
// CHECK-NEXT:        %[[VAL_166:.*]] = memref.alloca() : memref<1x!sycl_range_1_>
// CHECK-NEXT:        %[[VAL_167:.*]] = memref.alloca() : memref<1x!sycl_range_1_>
// CHECK-NEXT:        call @__init.1_k0_impl.1(%[[VAL_164]], %[[VAL_165]], %[[VAL_166]], %[[VAL_167]]) : (memref<1x!sycl_id_1_>, memref<1x!sycl_range_1_>, memref<1x!sycl_range_1_>, memref<1x!sycl_range_1_>) -> ()
// CHECK-NEXT:        %[[VAL_168:.*]] = sycl.id.get %[[VAL_164]]{{\[}}%[[VAL_163]]] : (memref<1x!sycl_id_1_>, i32) -> index
// CHECK-NEXT:        %[[VAL_169:.*]] = sycl.range.get %[[VAL_165]]{{\[}}%[[VAL_163]]] : (memref<1x!sycl_range_1_>, i32) -> index
// CHECK-NEXT:        %[[VAL_170:.*]] = sycl.range.get %[[VAL_166]]{{\[}}%[[VAL_163]]] : (memref<1x!sycl_range_1_>, i32) -> index
// CHECK-NEXT:        %[[VAL_171:.*]] = sycl.range.get %[[VAL_167]]{{\[}}%[[VAL_163]]] : (memref<1x!sycl_range_1_>, i32) -> index
// CHECK-NEXT:        memref.store %[[VAL_168]], %[[VAL_158]]{{\[}}%[[VAL_162]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_169]], %[[VAL_159]]{{\[}}%[[VAL_162]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_170]], %[[VAL_160]]{{\[}}%[[VAL_162]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_171]], %[[VAL_161]]{{\[}}%[[VAL_162]]] : memref<?xindex>
// CHECK-NEXT:        return
// CHECK-NEXT:      }

// CHECK-LABEL:     func.func @__impl.1_k1(
// CHECK-SAME:                             %[[VAL_172:.*]]: memref<?xindex>, %[[VAL_173:.*]]: memref<?xindex>, %[[VAL_174:.*]]: memref<?xindex>, %[[VAL_175:.*]]: memref<?xindex>) {
// CHECK-NEXT:        %[[VAL_176:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_177:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[VAL_178:.*]] = memref.alloca() : memref<1x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_179:.*]] = memref.alloca() : memref<1x!sycl_range_1_>
// CHECK-NEXT:        %[[VAL_180:.*]] = memref.alloca() : memref<1x!sycl_range_1_>
// CHECK-NEXT:        %[[VAL_181:.*]] = memref.alloca() : memref<1x!sycl_range_1_>
// CHECK-NEXT:        call @__init.1_k1_impl.1(%[[VAL_178]], %[[VAL_179]], %[[VAL_180]], %[[VAL_181]]) : (memref<1x!sycl_id_1_>, memref<1x!sycl_range_1_>, memref<1x!sycl_range_1_>, memref<1x!sycl_range_1_>) -> ()
// CHECK-NEXT:        %[[VAL_182:.*]] = sycl.id.get %[[VAL_178]]{{\[}}%[[VAL_177]]] : (memref<1x!sycl_id_1_>, i32) -> index
// CHECK-NEXT:        %[[VAL_183:.*]] = sycl.range.get %[[VAL_179]]{{\[}}%[[VAL_177]]] : (memref<1x!sycl_range_1_>, i32) -> index
// CHECK-NEXT:        %[[VAL_184:.*]] = sycl.range.get %[[VAL_180]]{{\[}}%[[VAL_177]]] : (memref<1x!sycl_range_1_>, i32) -> index
// CHECK-NEXT:        %[[VAL_185:.*]] = sycl.range.get %[[VAL_181]]{{\[}}%[[VAL_177]]] : (memref<1x!sycl_range_1_>, i32) -> index
// CHECK-NEXT:        memref.store %[[VAL_182]], %[[VAL_172]]{{\[}}%[[VAL_176]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_183]], %[[VAL_173]]{{\[}}%[[VAL_176]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_184]], %[[VAL_174]]{{\[}}%[[VAL_176]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_185]], %[[VAL_175]]{{\[}}%[[VAL_176]]] : memref<?xindex>
// CHECK-NEXT:        return
// CHECK-NEXT:      }
  func.func @impl.1(%res.offset: memref<?xindex>,
                    %res.gs: memref<?xindex>,
                    %res.ls: memref<?xindex>,
                    %res.nws: memref<?xindex>) {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %offset = memref.alloca() : memref<1x!sycl_id_1_>
    %global_size = memref.alloca() : memref<1x!sycl_range_1_>
    %local_size = memref.alloca() : memref<1x!sycl_range_1_>
    %num_work_groups = memref.alloca() : memref<1x!sycl_range_1_>
    func.call @init.1(%offset, %global_size, %local_size, %num_work_groups)
        : (memref<1x!sycl_id_1_>, memref<1x!sycl_range_1_>,
           memref<1x!sycl_range_1_>, memref<1x!sycl_range_1_>) -> ()
    %offset.0 = sycl.id.get %offset[%c0_i32]
        : (memref<1x!sycl_id_1_>, i32) -> index
    %global_size.0 = sycl.range.get %global_size[%c0_i32]
        : (memref<1x!sycl_range_1_>, i32) -> index
    %local_size.0 = sycl.range.get %local_size[%c0_i32]
        : (memref<1x!sycl_range_1_>, i32) -> index
    %num_work_groups.0 = sycl.range.get %num_work_groups[%c0_i32]
        : (memref<1x!sycl_range_1_>, i32) -> index
    memref.store %offset.0, %res.offset[%c0] : memref<?xindex>
    memref.store %global_size.0, %res.gs[%c0] : memref<?xindex>
    memref.store %local_size.0, %res.ls[%c0] : memref<?xindex>
    memref.store %num_work_groups.0, %res.nws[%c0] : memref<?xindex>
    func.return
  }

// CHECK-LABEL:     func.func @__impl.2_k2(
// CHECK-SAME:                             %[[VAL_200:.*]]: memref<?xindex>, %[[VAL_201:.*]]: memref<?xindex>, %[[VAL_202:.*]]: memref<?xindex>, %[[VAL_203:.*]]: memref<?xindex>, %[[VAL_204:.*]]: memref<?xindex>, %[[VAL_205:.*]]: memref<?xindex>, %[[VAL_206:.*]]: memref<?xindex>, %[[VAL_207:.*]]: memref<?xindex>) {
// CHECK-NEXT:        %[[VAL_208:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_209:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[VAL_210:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_211:.*]] = memref.alloca() : memref<1x!sycl_id_2_>
// CHECK-NEXT:        %[[VAL_212:.*]] = memref.alloca() : memref<1x!sycl_range_2_>
// CHECK-NEXT:        %[[VAL_213:.*]] = memref.alloca() : memref<1x!sycl_range_2_>
// CHECK-NEXT:        %[[VAL_214:.*]] = memref.alloca() : memref<1x!sycl_range_2_>
// CHECK-NEXT:        call @__init.2_k2_impl.2(%[[VAL_211]], %[[VAL_212]], %[[VAL_213]], %[[VAL_214]]) : (memref<1x!sycl_id_2_>, memref<1x!sycl_range_2_>, memref<1x!sycl_range_2_>, memref<1x!sycl_range_2_>) -> ()
// CHECK-NEXT:        %[[VAL_215:.*]] = sycl.id.get %[[VAL_211]]{{\[}}%[[VAL_209]]] : (memref<1x!sycl_id_2_>, i32) -> index
// CHECK-NEXT:        %[[VAL_216:.*]] = sycl.range.get %[[VAL_212]]{{\[}}%[[VAL_209]]] : (memref<1x!sycl_range_2_>, i32) -> index
// CHECK-NEXT:        %[[VAL_217:.*]] = sycl.range.get %[[VAL_213]]{{\[}}%[[VAL_209]]] : (memref<1x!sycl_range_2_>, i32) -> index
// CHECK-NEXT:        %[[VAL_218:.*]] = sycl.range.get %[[VAL_214]]{{\[}}%[[VAL_209]]] : (memref<1x!sycl_range_2_>, i32) -> index
// CHECK-NEXT:        %[[VAL_219:.*]] = sycl.id.get %[[VAL_211]]{{\[}}%[[VAL_210]]] : (memref<1x!sycl_id_2_>, i32) -> index
// CHECK-NEXT:        %[[VAL_220:.*]] = sycl.range.get %[[VAL_212]]{{\[}}%[[VAL_210]]] : (memref<1x!sycl_range_2_>, i32) -> index
// CHECK-NEXT:        %[[VAL_221:.*]] = sycl.range.get %[[VAL_213]]{{\[}}%[[VAL_210]]] : (memref<1x!sycl_range_2_>, i32) -> index
// CHECK-NEXT:        %[[VAL_222:.*]] = sycl.range.get %[[VAL_214]]{{\[}}%[[VAL_210]]] : (memref<1x!sycl_range_2_>, i32) -> index
// CHECK-NEXT:        memref.store %[[VAL_215]], %[[VAL_200]]{{\[}}%[[VAL_208]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_216]], %[[VAL_202]]{{\[}}%[[VAL_208]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_217]], %[[VAL_204]]{{\[}}%[[VAL_208]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_218]], %[[VAL_206]]{{\[}}%[[VAL_208]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_219]], %[[VAL_201]]{{\[}}%[[VAL_208]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_220]], %[[VAL_203]]{{\[}}%[[VAL_208]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_221]], %[[VAL_205]]{{\[}}%[[VAL_208]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_222]], %[[VAL_207]]{{\[}}%[[VAL_208]]] : memref<?xindex>
// CHECK-NEXT:        return
// CHECK-NEXT:      }
  func.func @impl.2(%res.offset.0: memref<?xindex>,
                    %res.offset.1: memref<?xindex>,
                    %res.gs.0: memref<?xindex>,
                    %res.gs.1: memref<?xindex>,
                    %res.ls.0: memref<?xindex>,
                    %res.ls.1: memref<?xindex>,
                    %res.nws.0: memref<?xindex>,
                    %res.nws.1: memref<?xindex>) {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %offset = memref.alloca() : memref<1x!sycl_id_2_>
    %global_size = memref.alloca() : memref<1x!sycl_range_2_>
    %local_size = memref.alloca() : memref<1x!sycl_range_2_>
    %num_work_groups = memref.alloca() : memref<1x!sycl_range_2_>
    func.call @init.2(%offset, %global_size, %local_size, %num_work_groups)
        : (memref<1x!sycl_id_2_>, memref<1x!sycl_range_2_>,
           memref<1x!sycl_range_2_>, memref<1x!sycl_range_2_>) -> ()
    %offset.0 = sycl.id.get %offset[%c0_i32]
        : (memref<1x!sycl_id_2_>, i32) -> index
    %global_size.0 = sycl.range.get %global_size[%c0_i32]
        : (memref<1x!sycl_range_2_>, i32) -> index
    %local_size.0 = sycl.range.get %local_size[%c0_i32]
        : (memref<1x!sycl_range_2_>, i32) -> index
    %num_work_groups.0 = sycl.range.get %num_work_groups[%c0_i32]
        : (memref<1x!sycl_range_2_>, i32) -> index
    %offset.1 = sycl.id.get %offset[%c1]
        : (memref<1x!sycl_id_2_>, i32) -> index
    %global_size.1 = sycl.range.get %global_size[%c1]
        : (memref<1x!sycl_range_2_>, i32) -> index
    %local_size.1 = sycl.range.get %local_size[%c1]
        : (memref<1x!sycl_range_2_>, i32) -> index
    %num_work_groups.1 = sycl.range.get %num_work_groups[%c1]
        : (memref<1x!sycl_range_2_>, i32) -> index
    memref.store %offset.0, %res.offset.0[%c0] : memref<?xindex>
    memref.store %global_size.0, %res.gs.0[%c0] : memref<?xindex>
    memref.store %local_size.0, %res.ls.0[%c0] : memref<?xindex>
    memref.store %num_work_groups.0, %res.nws.0[%c0] : memref<?xindex>
    memref.store %offset.1, %res.offset.1[%c0] : memref<?xindex>
    memref.store %global_size.1, %res.gs.1[%c0] : memref<?xindex>
    memref.store %local_size.1, %res.ls.1[%c0] : memref<?xindex>
    memref.store %num_work_groups.1, %res.nws.1[%c0] : memref<?xindex>
    func.return
  }

// CHECK-LABEL:     func.func @__impl.3_k3(
// CHECK-SAME:                             %[[VAL_246:.*]]: memref<?xindex>, %[[VAL_247:.*]]: memref<?xindex>, %[[VAL_248:.*]]: memref<?xindex>, %[[VAL_249:.*]]: memref<?xindex>, %[[VAL_250:.*]]: memref<?xindex>, %[[VAL_251:.*]]: memref<?xindex>, %[[VAL_252:.*]]: memref<?xindex>, %[[VAL_253:.*]]: memref<?xindex>, %[[VAL_254:.*]]: memref<?xindex>, %[[VAL_255:.*]]: memref<?xindex>, %[[VAL_256:.*]]: memref<?xindex>, %[[VAL_257:.*]]: memref<?xindex>) {
// CHECK-NEXT:        %[[VAL_258:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_259:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[VAL_260:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_261:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_262:.*]] = memref.alloca() : memref<1x!sycl_id_3_>
// CHECK-NEXT:        %[[VAL_263:.*]] = memref.alloca() : memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_264:.*]] = memref.alloca() : memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_265:.*]] = memref.alloca() : memref<1x!sycl_range_3_>
// CHECK-NEXT:        call @__init.3_k3_impl.3(%[[VAL_262]], %[[VAL_263]], %[[VAL_264]], %[[VAL_265]]) : (memref<1x!sycl_id_3_>, memref<1x!sycl_range_3_>, memref<1x!sycl_range_3_>, memref<1x!sycl_range_3_>) -> ()
// CHECK-NEXT:        %[[VAL_266:.*]] = sycl.id.get %[[VAL_262]]{{\[}}%[[VAL_259]]] : (memref<1x!sycl_id_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_267:.*]] = sycl.range.get %[[VAL_263]]{{\[}}%[[VAL_259]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_268:.*]] = sycl.range.get %[[VAL_264]]{{\[}}%[[VAL_259]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_269:.*]] = sycl.range.get %[[VAL_265]]{{\[}}%[[VAL_259]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_270:.*]] = sycl.id.get %[[VAL_262]]{{\[}}%[[VAL_260]]] : (memref<1x!sycl_id_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_271:.*]] = sycl.range.get %[[VAL_263]]{{\[}}%[[VAL_260]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_272:.*]] = sycl.range.get %[[VAL_264]]{{\[}}%[[VAL_260]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_273:.*]] = sycl.range.get %[[VAL_265]]{{\[}}%[[VAL_260]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_274:.*]] = sycl.id.get %[[VAL_262]]{{\[}}%[[VAL_261]]] : (memref<1x!sycl_id_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_275:.*]] = sycl.range.get %[[VAL_263]]{{\[}}%[[VAL_261]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_276:.*]] = sycl.range.get %[[VAL_264]]{{\[}}%[[VAL_261]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_277:.*]] = sycl.range.get %[[VAL_265]]{{\[}}%[[VAL_261]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        memref.store %[[VAL_266]], %[[VAL_246]]{{\[}}%[[VAL_258]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_267]], %[[VAL_249]]{{\[}}%[[VAL_258]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_268]], %[[VAL_252]]{{\[}}%[[VAL_258]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_269]], %[[VAL_255]]{{\[}}%[[VAL_258]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_270]], %[[VAL_247]]{{\[}}%[[VAL_258]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_271]], %[[VAL_250]]{{\[}}%[[VAL_258]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_272]], %[[VAL_253]]{{\[}}%[[VAL_258]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_273]], %[[VAL_256]]{{\[}}%[[VAL_258]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_274]], %[[VAL_248]]{{\[}}%[[VAL_258]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_275]], %[[VAL_251]]{{\[}}%[[VAL_258]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_276]], %[[VAL_254]]{{\[}}%[[VAL_258]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_277]], %[[VAL_257]]{{\[}}%[[VAL_258]]] : memref<?xindex>
// CHECK-NEXT:        return
// CHECK-NEXT:      }

// CHECK-LABEL:     func.func @__impl.3_k4(
// CHECK-SAME:                             %[[VAL_278:.*]]: memref<?xindex>, %[[VAL_279:.*]]: memref<?xindex>, %[[VAL_280:.*]]: memref<?xindex>, %[[VAL_281:.*]]: memref<?xindex>, %[[VAL_282:.*]]: memref<?xindex>, %[[VAL_283:.*]]: memref<?xindex>, %[[VAL_284:.*]]: memref<?xindex>, %[[VAL_285:.*]]: memref<?xindex>, %[[VAL_286:.*]]: memref<?xindex>, %[[VAL_287:.*]]: memref<?xindex>, %[[VAL_288:.*]]: memref<?xindex>, %[[VAL_289:.*]]: memref<?xindex>) {
// CHECK-NEXT:        %[[VAL_290:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_291:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[VAL_292:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_293:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_294:.*]] = memref.alloca() : memref<1x!sycl_id_3_>
// CHECK-NEXT:        %[[VAL_295:.*]] = memref.alloca() : memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_296:.*]] = memref.alloca() : memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_297:.*]] = memref.alloca() : memref<1x!sycl_range_3_>
// CHECK-NEXT:        call @__init.3_k4_impl.3(%[[VAL_294]], %[[VAL_295]], %[[VAL_296]], %[[VAL_297]]) : (memref<1x!sycl_id_3_>, memref<1x!sycl_range_3_>, memref<1x!sycl_range_3_>, memref<1x!sycl_range_3_>) -> ()
// CHECK-NEXT:        %[[VAL_298:.*]] = sycl.id.get %[[VAL_294]]{{\[}}%[[VAL_291]]] : (memref<1x!sycl_id_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_299:.*]] = sycl.range.get %[[VAL_295]]{{\[}}%[[VAL_291]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_300:.*]] = sycl.range.get %[[VAL_296]]{{\[}}%[[VAL_291]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_301:.*]] = sycl.range.get %[[VAL_297]]{{\[}}%[[VAL_291]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_302:.*]] = sycl.id.get %[[VAL_294]]{{\[}}%[[VAL_292]]] : (memref<1x!sycl_id_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_303:.*]] = sycl.range.get %[[VAL_295]]{{\[}}%[[VAL_292]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_304:.*]] = sycl.range.get %[[VAL_296]]{{\[}}%[[VAL_292]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_305:.*]] = sycl.range.get %[[VAL_297]]{{\[}}%[[VAL_292]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_306:.*]] = sycl.id.get %[[VAL_294]]{{\[}}%[[VAL_293]]] : (memref<1x!sycl_id_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_307:.*]] = sycl.range.get %[[VAL_295]]{{\[}}%[[VAL_293]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_308:.*]] = sycl.range.get %[[VAL_296]]{{\[}}%[[VAL_293]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_309:.*]] = sycl.range.get %[[VAL_297]]{{\[}}%[[VAL_293]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        memref.store %[[VAL_298]], %[[VAL_278]]{{\[}}%[[VAL_290]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_299]], %[[VAL_281]]{{\[}}%[[VAL_290]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_300]], %[[VAL_284]]{{\[}}%[[VAL_290]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_301]], %[[VAL_287]]{{\[}}%[[VAL_290]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_302]], %[[VAL_279]]{{\[}}%[[VAL_290]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_303]], %[[VAL_282]]{{\[}}%[[VAL_290]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_304]], %[[VAL_285]]{{\[}}%[[VAL_290]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_305]], %[[VAL_288]]{{\[}}%[[VAL_290]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_306]], %[[VAL_280]]{{\[}}%[[VAL_290]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_307]], %[[VAL_283]]{{\[}}%[[VAL_290]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_308]], %[[VAL_286]]{{\[}}%[[VAL_290]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_309]], %[[VAL_289]]{{\[}}%[[VAL_290]]] : memref<?xindex>
// CHECK-NEXT:        return
// CHECK-NEXT:      }

// CHECK-LABEL:     func.func @__impl.3_k5(
// CHECK-SAME:                             %[[VAL_310:.*]]: memref<?xindex>, %[[VAL_311:.*]]: memref<?xindex>, %[[VAL_312:.*]]: memref<?xindex>, %[[VAL_313:.*]]: memref<?xindex>, %[[VAL_314:.*]]: memref<?xindex>, %[[VAL_315:.*]]: memref<?xindex>, %[[VAL_316:.*]]: memref<?xindex>, %[[VAL_317:.*]]: memref<?xindex>, %[[VAL_318:.*]]: memref<?xindex>, %[[VAL_319:.*]]: memref<?xindex>, %[[VAL_320:.*]]: memref<?xindex>, %[[VAL_321:.*]]: memref<?xindex>) {
// CHECK-NEXT:        %[[VAL_322:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_323:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[VAL_324:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_325:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_326:.*]] = memref.alloca() : memref<1x!sycl_id_3_>
// CHECK-NEXT:        %[[VAL_327:.*]] = memref.alloca() : memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_328:.*]] = memref.alloca() : memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_329:.*]] = memref.alloca() : memref<1x!sycl_range_3_>
// CHECK-NEXT:        call @__init.3_k5_impl.3(%[[VAL_326]], %[[VAL_327]], %[[VAL_328]], %[[VAL_329]]) : (memref<1x!sycl_id_3_>, memref<1x!sycl_range_3_>, memref<1x!sycl_range_3_>, memref<1x!sycl_range_3_>) -> ()
// CHECK-NEXT:        %[[VAL_330:.*]] = sycl.id.get %[[VAL_326]]{{\[}}%[[VAL_323]]] : (memref<1x!sycl_id_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_331:.*]] = sycl.range.get %[[VAL_327]]{{\[}}%[[VAL_323]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_332:.*]] = sycl.range.get %[[VAL_328]]{{\[}}%[[VAL_323]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_333:.*]] = sycl.range.get %[[VAL_329]]{{\[}}%[[VAL_323]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_334:.*]] = sycl.id.get %[[VAL_326]]{{\[}}%[[VAL_324]]] : (memref<1x!sycl_id_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_335:.*]] = sycl.range.get %[[VAL_327]]{{\[}}%[[VAL_324]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_336:.*]] = sycl.range.get %[[VAL_328]]{{\[}}%[[VAL_324]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_337:.*]] = sycl.range.get %[[VAL_329]]{{\[}}%[[VAL_324]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_338:.*]] = sycl.id.get %[[VAL_326]]{{\[}}%[[VAL_325]]] : (memref<1x!sycl_id_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_339:.*]] = sycl.range.get %[[VAL_327]]{{\[}}%[[VAL_325]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_340:.*]] = sycl.range.get %[[VAL_328]]{{\[}}%[[VAL_325]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_341:.*]] = sycl.range.get %[[VAL_329]]{{\[}}%[[VAL_325]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        memref.store %[[VAL_330]], %[[VAL_310]]{{\[}}%[[VAL_322]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_331]], %[[VAL_313]]{{\[}}%[[VAL_322]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_332]], %[[VAL_316]]{{\[}}%[[VAL_322]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_333]], %[[VAL_319]]{{\[}}%[[VAL_322]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_334]], %[[VAL_311]]{{\[}}%[[VAL_322]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_335]], %[[VAL_314]]{{\[}}%[[VAL_322]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_336]], %[[VAL_317]]{{\[}}%[[VAL_322]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_337]], %[[VAL_320]]{{\[}}%[[VAL_322]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_338]], %[[VAL_312]]{{\[}}%[[VAL_322]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_339]], %[[VAL_315]]{{\[}}%[[VAL_322]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_340]], %[[VAL_318]]{{\[}}%[[VAL_322]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_341]], %[[VAL_321]]{{\[}}%[[VAL_322]]] : memref<?xindex>
// CHECK-NEXT:        return
// CHECK-NEXT:      }

// CHECK-LABEL:     func.func @__impl.3_k6(
// CHECK-SAME:                             %[[VAL_342:.*]]: memref<?xindex>, %[[VAL_343:.*]]: memref<?xindex>, %[[VAL_344:.*]]: memref<?xindex>, %[[VAL_345:.*]]: memref<?xindex>, %[[VAL_346:.*]]: memref<?xindex>, %[[VAL_347:.*]]: memref<?xindex>, %[[VAL_348:.*]]: memref<?xindex>, %[[VAL_349:.*]]: memref<?xindex>, %[[VAL_350:.*]]: memref<?xindex>, %[[VAL_351:.*]]: memref<?xindex>, %[[VAL_352:.*]]: memref<?xindex>, %[[VAL_353:.*]]: memref<?xindex>) {
// CHECK-NEXT:        %[[VAL_354:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_355:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[VAL_356:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_357:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_358:.*]] = memref.alloca() : memref<1x!sycl_id_3_>
// CHECK-NEXT:        %[[VAL_359:.*]] = memref.alloca() : memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_360:.*]] = memref.alloca() : memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_361:.*]] = memref.alloca() : memref<1x!sycl_range_3_>
// CHECK-NEXT:        call @__init.3_k6_impl.3(%[[VAL_358]], %[[VAL_359]], %[[VAL_360]], %[[VAL_361]]) : (memref<1x!sycl_id_3_>, memref<1x!sycl_range_3_>, memref<1x!sycl_range_3_>, memref<1x!sycl_range_3_>) -> ()
// CHECK-NEXT:        %[[VAL_362:.*]] = sycl.id.get %[[VAL_358]]{{\[}}%[[VAL_355]]] : (memref<1x!sycl_id_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_363:.*]] = sycl.range.get %[[VAL_359]]{{\[}}%[[VAL_355]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_364:.*]] = sycl.range.get %[[VAL_360]]{{\[}}%[[VAL_355]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_365:.*]] = sycl.range.get %[[VAL_361]]{{\[}}%[[VAL_355]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_366:.*]] = sycl.id.get %[[VAL_358]]{{\[}}%[[VAL_356]]] : (memref<1x!sycl_id_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_367:.*]] = sycl.range.get %[[VAL_359]]{{\[}}%[[VAL_356]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_368:.*]] = sycl.range.get %[[VAL_360]]{{\[}}%[[VAL_356]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_369:.*]] = sycl.range.get %[[VAL_361]]{{\[}}%[[VAL_356]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_370:.*]] = sycl.id.get %[[VAL_358]]{{\[}}%[[VAL_357]]] : (memref<1x!sycl_id_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_371:.*]] = sycl.range.get %[[VAL_359]]{{\[}}%[[VAL_357]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_372:.*]] = sycl.range.get %[[VAL_360]]{{\[}}%[[VAL_357]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_373:.*]] = sycl.range.get %[[VAL_361]]{{\[}}%[[VAL_357]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        memref.store %[[VAL_362]], %[[VAL_342]]{{\[}}%[[VAL_354]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_363]], %[[VAL_345]]{{\[}}%[[VAL_354]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_364]], %[[VAL_348]]{{\[}}%[[VAL_354]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_365]], %[[VAL_351]]{{\[}}%[[VAL_354]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_366]], %[[VAL_343]]{{\[}}%[[VAL_354]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_367]], %[[VAL_346]]{{\[}}%[[VAL_354]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_368]], %[[VAL_349]]{{\[}}%[[VAL_354]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_369]], %[[VAL_352]]{{\[}}%[[VAL_354]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_370]], %[[VAL_344]]{{\[}}%[[VAL_354]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_371]], %[[VAL_347]]{{\[}}%[[VAL_354]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_372]], %[[VAL_350]]{{\[}}%[[VAL_354]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_373]], %[[VAL_353]]{{\[}}%[[VAL_354]]] : memref<?xindex>
// CHECK-NEXT:        return
// CHECK-NEXT:      }

// CHECK-LABEL:     func.func @__impl.3_k7(
// CHECK-SAME:                             %[[VAL_374:.*]]: memref<?xindex>, %[[VAL_375:.*]]: memref<?xindex>, %[[VAL_376:.*]]: memref<?xindex>, %[[VAL_377:.*]]: memref<?xindex>, %[[VAL_378:.*]]: memref<?xindex>, %[[VAL_379:.*]]: memref<?xindex>, %[[VAL_380:.*]]: memref<?xindex>, %[[VAL_381:.*]]: memref<?xindex>, %[[VAL_382:.*]]: memref<?xindex>, %[[VAL_383:.*]]: memref<?xindex>, %[[VAL_384:.*]]: memref<?xindex>, %[[VAL_385:.*]]: memref<?xindex>) {
// CHECK-NEXT:        %[[VAL_386:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_387:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[VAL_388:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_389:.*]] = arith.constant 1 : i32
// CHECK-NEXT:        %[[VAL_390:.*]] = memref.alloca() : memref<1x!sycl_id_3_>
// CHECK-NEXT:        %[[VAL_391:.*]] = memref.alloca() : memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_392:.*]] = memref.alloca() : memref<1x!sycl_range_3_>
// CHECK-NEXT:        %[[VAL_393:.*]] = memref.alloca() : memref<1x!sycl_range_3_>
// CHECK-NEXT:        call @__init.3_k7_impl.3(%[[VAL_390]], %[[VAL_391]], %[[VAL_392]], %[[VAL_393]]) : (memref<1x!sycl_id_3_>, memref<1x!sycl_range_3_>, memref<1x!sycl_range_3_>, memref<1x!sycl_range_3_>) -> ()
// CHECK-NEXT:        %[[VAL_394:.*]] = sycl.id.get %[[VAL_390]]{{\[}}%[[VAL_387]]] : (memref<1x!sycl_id_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_395:.*]] = sycl.range.get %[[VAL_391]]{{\[}}%[[VAL_387]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_396:.*]] = sycl.range.get %[[VAL_392]]{{\[}}%[[VAL_387]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_397:.*]] = sycl.range.get %[[VAL_393]]{{\[}}%[[VAL_387]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_398:.*]] = sycl.id.get %[[VAL_390]]{{\[}}%[[VAL_388]]] : (memref<1x!sycl_id_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_399:.*]] = sycl.range.get %[[VAL_391]]{{\[}}%[[VAL_388]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_400:.*]] = sycl.range.get %[[VAL_392]]{{\[}}%[[VAL_388]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_401:.*]] = sycl.range.get %[[VAL_393]]{{\[}}%[[VAL_388]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_402:.*]] = sycl.id.get %[[VAL_390]]{{\[}}%[[VAL_389]]] : (memref<1x!sycl_id_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_403:.*]] = sycl.range.get %[[VAL_391]]{{\[}}%[[VAL_389]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_404:.*]] = sycl.range.get %[[VAL_392]]{{\[}}%[[VAL_389]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        %[[VAL_405:.*]] = sycl.range.get %[[VAL_393]]{{\[}}%[[VAL_389]]] : (memref<1x!sycl_range_3_>, i32) -> index
// CHECK-NEXT:        memref.store %[[VAL_394]], %[[VAL_374]]{{\[}}%[[VAL_386]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_395]], %[[VAL_377]]{{\[}}%[[VAL_386]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_396]], %[[VAL_380]]{{\[}}%[[VAL_386]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_397]], %[[VAL_383]]{{\[}}%[[VAL_386]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_398]], %[[VAL_375]]{{\[}}%[[VAL_386]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_399]], %[[VAL_378]]{{\[}}%[[VAL_386]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_400]], %[[VAL_381]]{{\[}}%[[VAL_386]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_401]], %[[VAL_384]]{{\[}}%[[VAL_386]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_402]], %[[VAL_376]]{{\[}}%[[VAL_386]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_403]], %[[VAL_379]]{{\[}}%[[VAL_386]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_404]], %[[VAL_382]]{{\[}}%[[VAL_386]]] : memref<?xindex>
// CHECK-NEXT:        memref.store %[[VAL_405]], %[[VAL_385]]{{\[}}%[[VAL_386]]] : memref<?xindex>
// CHECK-NEXT:        return
// CHECK-NEXT:      }
  func.func @impl.3(%res.offset.0: memref<?xindex>,
                    %res.offset.1: memref<?xindex>,
                    %res.offset.2: memref<?xindex>,
                    %res.gs.0: memref<?xindex>,
                    %res.gs.1: memref<?xindex>,
                    %res.gs.2: memref<?xindex>,
                    %res.ls.0: memref<?xindex>,
                    %res.ls.1: memref<?xindex>,
                    %res.ls.2: memref<?xindex>,
                    %res.nws.0: memref<?xindex>,
                    %res.nws.1: memref<?xindex>,
                    %res.nws.2: memref<?xindex>) {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 1 : i32
    %offset = memref.alloca() : memref<1x!sycl_id_3_>
    %global_size = memref.alloca() : memref<1x!sycl_range_3_>
    %local_size = memref.alloca() : memref<1x!sycl_range_3_>
    %num_work_groups = memref.alloca() : memref<1x!sycl_range_3_>
    func.call @init.3(%offset, %global_size, %local_size, %num_work_groups)
        : (memref<1x!sycl_id_3_>, memref<1x!sycl_range_3_>,
           memref<1x!sycl_range_3_>, memref<1x!sycl_range_3_>) -> ()
    %offset.0 = sycl.id.get %offset[%c0_i32]
        : (memref<1x!sycl_id_3_>, i32) -> index
    %global_size.0 = sycl.range.get %global_size[%c0_i32]
        : (memref<1x!sycl_range_3_>, i32) -> index
    %local_size.0 = sycl.range.get %local_size[%c0_i32]
        : (memref<1x!sycl_range_3_>, i32) -> index
    %num_work_groups.0 = sycl.range.get %num_work_groups[%c0_i32]
        : (memref<1x!sycl_range_3_>, i32) -> index
    %offset.1 = sycl.id.get %offset[%c1]
        : (memref<1x!sycl_id_3_>, i32) -> index
    %global_size.1 = sycl.range.get %global_size[%c1]
        : (memref<1x!sycl_range_3_>, i32) -> index
    %local_size.1 = sycl.range.get %local_size[%c1]
        : (memref<1x!sycl_range_3_>, i32) -> index
    %num_work_groups.1 = sycl.range.get %num_work_groups[%c1]
        : (memref<1x!sycl_range_3_>, i32) -> index
    %offset.2 = sycl.id.get %offset[%c2]
        : (memref<1x!sycl_id_3_>, i32) -> index
    %global_size.2 = sycl.range.get %global_size[%c2]
        : (memref<1x!sycl_range_3_>, i32) -> index
    %local_size.2 = sycl.range.get %local_size[%c2]
        : (memref<1x!sycl_range_3_>, i32) -> index
    %num_work_groups.2 = sycl.range.get %num_work_groups[%c2]
        : (memref<1x!sycl_range_3_>, i32) -> index
    memref.store %offset.0, %res.offset.0[%c0] : memref<?xindex>
    memref.store %global_size.0, %res.gs.0[%c0] : memref<?xindex>
    memref.store %local_size.0, %res.ls.0[%c0] : memref<?xindex>
    memref.store %num_work_groups.0, %res.nws.0[%c0] : memref<?xindex>
    memref.store %offset.1, %res.offset.1[%c0] : memref<?xindex>
    memref.store %global_size.1, %res.gs.1[%c0] : memref<?xindex>
    memref.store %local_size.1, %res.ls.1[%c0] : memref<?xindex>
    memref.store %num_work_groups.1, %res.nws.1[%c0] : memref<?xindex>
    memref.store %offset.2, %res.offset.2[%c0] : memref<?xindex>
    memref.store %global_size.2, %res.gs.2[%c0] : memref<?xindex>
    memref.store %local_size.2, %res.ls.2[%c0] : memref<?xindex>
    memref.store %num_work_groups.2, %res.nws.2[%c0] : memref<?xindex>
    func.return
  }

// CHECK-LABEL:     gpu.func @k0(
// CHECK-SAME:                   %[[VAL_438:.*]]: memref<?xindex>, %[[VAL_439:.*]]: memref<?xindex>, %[[VAL_440:.*]]: memref<?xindex>, %[[VAL_441:.*]]: memref<?xindex>) kernel {
// CHECK-NEXT:        func.call @__impl.1_k0(%[[VAL_438]], %[[VAL_439]], %[[VAL_440]], %[[VAL_441]]) : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>) -> ()
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }
  gpu.func @k0(%res.offset: memref<?xindex>,
               %res.gs: memref<?xindex>,
               %res.ls: memref<?xindex>,
               %res.nws: memref<?xindex>) kernel {
    func.call @impl.1(%res.offset, %res.gs, %res.ls, %res.nws)
        : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>)
        -> ()
    gpu.return
  }

// CHECK-LABEL:     gpu.func @k1(
// CHECK-SAME:                   %[[VAL_442:.*]]: memref<?xindex>, %[[VAL_443:.*]]: memref<?xindex>, %[[VAL_444:.*]]: memref<?xindex>, %[[VAL_445:.*]]: memref<?xindex>) kernel {
// CHECK-NEXT:        func.call @__impl.1_k1(%[[VAL_442]], %[[VAL_443]], %[[VAL_444]], %[[VAL_445]]) : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>) -> ()
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }
  gpu.func @k1(%res.offset: memref<?xindex>,
               %res.gs: memref<?xindex>,
               %res.ls: memref<?xindex>,
               %res.nws: memref<?xindex>) kernel {
    func.call @impl.1(%res.offset, %res.gs, %res.ls, %res.nws)
        : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>) -> ()
    gpu.return
  }

// CHECK-LABEL:     gpu.func @k2(
// CHECK-SAME:                   %[[VAL_446:.*]]: memref<?xindex>, %[[VAL_447:.*]]: memref<?xindex>, %[[VAL_448:.*]]: memref<?xindex>, %[[VAL_449:.*]]: memref<?xindex>, %[[VAL_450:.*]]: memref<?xindex>, %[[VAL_451:.*]]: memref<?xindex>, %[[VAL_452:.*]]: memref<?xindex>, %[[VAL_453:.*]]: memref<?xindex>) kernel {
// CHECK-NEXT:        func.call @__impl.2_k2(%[[VAL_446]], %[[VAL_447]], %[[VAL_448]], %[[VAL_449]], %[[VAL_450]], %[[VAL_451]], %[[VAL_452]], %[[VAL_453]]) : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>) -> ()
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }
  gpu.func @k2(%res.offset.0: memref<?xindex>,
               %res.offset.1: memref<?xindex>,
               %res.gs.0: memref<?xindex>,
               %res.gs.1: memref<?xindex>,
               %res.ls.0: memref<?xindex>,
               %res.ls.1: memref<?xindex>,
               %res.nws.0: memref<?xindex>,
               %res.nws.1: memref<?xindex>) kernel {
    func.call @impl.2(%res.offset.0,
                      %res.offset.1,
                      %res.gs.0,
                      %res.gs.1,
                      %res.ls.0,
                      %res.ls.1,
                      %res.nws.0,
                      %res.nws.1)
        : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>,
           memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>)
         -> ()
    gpu.return
  }

// CHECK-LABEL:     gpu.func @k3(
// CHECK-SAME:                   %[[VAL_454:.*]]: memref<?xindex>, %[[VAL_455:.*]]: memref<?xindex>, %[[VAL_456:.*]]: memref<?xindex>, %[[VAL_457:.*]]: memref<?xindex>, %[[VAL_458:.*]]: memref<?xindex>, %[[VAL_459:.*]]: memref<?xindex>, %[[VAL_460:.*]]: memref<?xindex>, %[[VAL_461:.*]]: memref<?xindex>, %[[VAL_462:.*]]: memref<?xindex>, %[[VAL_463:.*]]: memref<?xindex>, %[[VAL_464:.*]]: memref<?xindex>, %[[VAL_465:.*]]: memref<?xindex>) kernel {
// CHECK-NEXT:        func.call @__impl.3_k3(%[[VAL_454]], %[[VAL_455]], %[[VAL_456]], %[[VAL_457]], %[[VAL_458]], %[[VAL_459]], %[[VAL_460]], %[[VAL_461]], %[[VAL_462]], %[[VAL_463]], %[[VAL_464]], %[[VAL_465]]) : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>) -> ()
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }
  gpu.func @k3(%res.offset.0: memref<?xindex>,
               %res.offset.1: memref<?xindex>,
               %res.offset.2: memref<?xindex>,
               %res.gs.0: memref<?xindex>,
               %res.gs.1: memref<?xindex>,
               %res.gs.2: memref<?xindex>,
               %res.ls.0: memref<?xindex>,
               %res.ls.1: memref<?xindex>,
               %res.ls.2: memref<?xindex>,
               %res.nws.0: memref<?xindex>,
               %res.nws.1: memref<?xindex>,
               %res.nws.2: memref<?xindex>) kernel {
    func.call @impl.3(%res.offset.0,
                      %res.offset.1,
                      %res.offset.2,
                      %res.gs.0,
                      %res.gs.1,
                      %res.gs.2,
                      %res.ls.0,
                      %res.ls.1,
                      %res.ls.2,
                      %res.nws.0,
                      %res.nws.1,
                      %res.nws.2)
        : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>,
           memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>,
           memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>)
        -> ()
    gpu.return
  }

// CHECK-LABEL:     gpu.func @k4(
// CHECK-SAME:                   %[[VAL_466:.*]]: memref<?xindex>, %[[VAL_467:.*]]: memref<?xindex>, %[[VAL_468:.*]]: memref<?xindex>, %[[VAL_469:.*]]: memref<?xindex>, %[[VAL_470:.*]]: memref<?xindex>, %[[VAL_471:.*]]: memref<?xindex>, %[[VAL_472:.*]]: memref<?xindex>, %[[VAL_473:.*]]: memref<?xindex>, %[[VAL_474:.*]]: memref<?xindex>, %[[VAL_475:.*]]: memref<?xindex>, %[[VAL_476:.*]]: memref<?xindex>, %[[VAL_477:.*]]: memref<?xindex>) kernel {
// CHECK-NEXT:        func.call @__impl.3_k4(%[[VAL_466]], %[[VAL_467]], %[[VAL_468]], %[[VAL_469]], %[[VAL_470]], %[[VAL_471]], %[[VAL_472]], %[[VAL_473]], %[[VAL_474]], %[[VAL_475]], %[[VAL_476]], %[[VAL_477]]) : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>) -> ()
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }
  gpu.func @k4(%res.offset.0: memref<?xindex>,
               %res.offset.1: memref<?xindex>,
               %res.offset.2: memref<?xindex>,
               %res.gs.0: memref<?xindex>,
               %res.gs.1: memref<?xindex>,
               %res.gs.2: memref<?xindex>,
               %res.ls.0: memref<?xindex>,
               %res.ls.1: memref<?xindex>,
               %res.ls.2: memref<?xindex>,
               %res.nws.0: memref<?xindex>,
               %res.nws.1: memref<?xindex>,
               %res.nws.2: memref<?xindex>) kernel {
    func.call @impl.3(%res.offset.0,
                      %res.offset.1,
                      %res.offset.2,
                      %res.gs.0,
                      %res.gs.1,
                      %res.gs.2,
                      %res.ls.0,
                      %res.ls.1,
                      %res.ls.2,
                      %res.nws.0,
                      %res.nws.1,
                      %res.nws.2)
        : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>,
           memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>,
           memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>)
        -> ()
    gpu.return
  }

// CHECK-LABEL:     gpu.func @k5(
// CHECK-SAME:                   %[[VAL_478:.*]]: memref<?xindex>, %[[VAL_479:.*]]: memref<?xindex>, %[[VAL_480:.*]]: memref<?xindex>, %[[VAL_481:.*]]: memref<?xindex>, %[[VAL_482:.*]]: memref<?xindex>, %[[VAL_483:.*]]: memref<?xindex>, %[[VAL_484:.*]]: memref<?xindex>, %[[VAL_485:.*]]: memref<?xindex>, %[[VAL_486:.*]]: memref<?xindex>, %[[VAL_487:.*]]: memref<?xindex>, %[[VAL_488:.*]]: memref<?xindex>, %[[VAL_489:.*]]: memref<?xindex>) kernel {
// CHECK-NEXT:        func.call @__impl.3_k5(%[[VAL_478]], %[[VAL_479]], %[[VAL_480]], %[[VAL_481]], %[[VAL_482]], %[[VAL_483]], %[[VAL_484]], %[[VAL_485]], %[[VAL_486]], %[[VAL_487]], %[[VAL_488]], %[[VAL_489]]) : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>) -> ()
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }
  gpu.func @k5(%res.offset.0: memref<?xindex>,
               %res.offset.1: memref<?xindex>,
               %res.offset.2: memref<?xindex>,
               %res.gs.0: memref<?xindex>,
               %res.gs.1: memref<?xindex>,
               %res.gs.2: memref<?xindex>,
               %res.ls.0: memref<?xindex>,
               %res.ls.1: memref<?xindex>,
               %res.ls.2: memref<?xindex>,
               %res.nws.0: memref<?xindex>,
               %res.nws.1: memref<?xindex>,
               %res.nws.2: memref<?xindex>) kernel {
    func.call @impl.3(%res.offset.0,
                      %res.offset.1,
                      %res.offset.2,
                      %res.gs.0,
                      %res.gs.1,
                      %res.gs.2,
                      %res.ls.0,
                      %res.ls.1,
                      %res.ls.2,
                      %res.nws.0,
                      %res.nws.1,
                      %res.nws.2)
        : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>,
           memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>,
           memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>)
        -> ()
    gpu.return
  }

// CHECK-LABEL:     gpu.func @k6(
// CHECK-SAME:                   %[[VAL_490:.*]]: memref<?xindex>, %[[VAL_491:.*]]: memref<?xindex>, %[[VAL_492:.*]]: memref<?xindex>, %[[VAL_493:.*]]: memref<?xindex>, %[[VAL_494:.*]]: memref<?xindex>, %[[VAL_495:.*]]: memref<?xindex>, %[[VAL_496:.*]]: memref<?xindex>, %[[VAL_497:.*]]: memref<?xindex>, %[[VAL_498:.*]]: memref<?xindex>, %[[VAL_499:.*]]: memref<?xindex>, %[[VAL_500:.*]]: memref<?xindex>, %[[VAL_501:.*]]: memref<?xindex>) kernel {
// CHECK-NEXT:        func.call @__impl.3_k6(%[[VAL_490]], %[[VAL_491]], %[[VAL_492]], %[[VAL_493]], %[[VAL_494]], %[[VAL_495]], %[[VAL_496]], %[[VAL_497]], %[[VAL_498]], %[[VAL_499]], %[[VAL_500]], %[[VAL_501]]) : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>) -> ()
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }
  gpu.func @k6(%res.offset.0: memref<?xindex>,
               %res.offset.1: memref<?xindex>,
               %res.offset.2: memref<?xindex>,
               %res.gs.0: memref<?xindex>,
               %res.gs.1: memref<?xindex>,
               %res.gs.2: memref<?xindex>,
               %res.ls.0: memref<?xindex>,
               %res.ls.1: memref<?xindex>,
               %res.ls.2: memref<?xindex>,
               %res.nws.0: memref<?xindex>,
               %res.nws.1: memref<?xindex>,
               %res.nws.2: memref<?xindex>) kernel {
    func.call @impl.3(%res.offset.0,
                      %res.offset.1,
                      %res.offset.2,
                      %res.gs.0,
                      %res.gs.1,
                      %res.gs.2,
                      %res.ls.0,
                      %res.ls.1,
                      %res.ls.2,
                      %res.nws.0,
                      %res.nws.1,
                      %res.nws.2)
        : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>,
           memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>,
           memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>)
        -> ()
    gpu.return
  }

// CHECK-LABEL:     gpu.func @k7(
// CHECK-SAME:                   %[[VAL_502:.*]]: memref<?xindex>, %[[VAL_503:.*]]: memref<?xindex>, %[[VAL_504:.*]]: memref<?xindex>, %[[VAL_505:.*]]: memref<?xindex>, %[[VAL_506:.*]]: memref<?xindex>, %[[VAL_507:.*]]: memref<?xindex>, %[[VAL_508:.*]]: memref<?xindex>, %[[VAL_509:.*]]: memref<?xindex>, %[[VAL_510:.*]]: memref<?xindex>, %[[VAL_511:.*]]: memref<?xindex>, %[[VAL_512:.*]]: memref<?xindex>, %[[VAL_513:.*]]: memref<?xindex>) kernel {
// CHECK-NEXT:        func.call @__impl.3_k7(%[[VAL_502]], %[[VAL_503]], %[[VAL_504]], %[[VAL_505]], %[[VAL_506]], %[[VAL_507]], %[[VAL_508]], %[[VAL_509]], %[[VAL_510]], %[[VAL_511]], %[[VAL_512]], %[[VAL_513]]) : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>) -> ()
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }
  gpu.func @k7(%res.offset.0: memref<?xindex>,
               %res.offset.1: memref<?xindex>,
               %res.offset.2: memref<?xindex>,
               %res.gs.0: memref<?xindex>,
               %res.gs.1: memref<?xindex>,
               %res.gs.2: memref<?xindex>,
               %res.ls.0: memref<?xindex>,
               %res.ls.1: memref<?xindex>,
               %res.ls.2: memref<?xindex>,
               %res.nws.0: memref<?xindex>,
               %res.nws.1: memref<?xindex>,
               %res.nws.2: memref<?xindex>) kernel {
    func.call @impl.3(%res.offset.0,
                      %res.offset.1,
                      %res.offset.2,
                      %res.gs.0,
                      %res.gs.1,
                      %res.gs.2,
                      %res.ls.0,
                      %res.ls.1,
                      %res.ls.2,
                      %res.nws.0,
                      %res.nws.1,
                      %res.nws.2)
        : (memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>,
           memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>,
           memref<?xindex>, memref<?xindex>, memref<?xindex>, memref<?xindex>)
        -> ()
    gpu.return
  }
}

// COM: Check we can detect the offset is constant (all-zeroes)

llvm.func internal @foo_default_offset(
    %handler: !llvm.ptr,
    %range.0: i64,
    %res.offset: !llvm.ptr, %res.gs: !llvm.ptr,
    %res.ls: !llvm.ptr, %res.nws: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %range = llvm.alloca %c1 x !llvm.struct<"sycl::_V1::range", opaque>
      : (i32) -> !llvm.ptr
  sycl.host.constructor(%range, %range.0) {type=!sycl_range_1_}
      : (!llvm.ptr, i64) -> ()
  sycl.host.handler.set_nd_range %handler -> range %range : !llvm.ptr, !llvm.ptr
  sycl.host.schedule_kernel %handler -> @kernels::@k0[range %range](
      %res.offset, %res.gs, %res.ls, %res.nws)
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// COM: Check we can detect the range is constant

llvm.func internal @foo_constant_range(
    %handler: !llvm.ptr,
    %offset.0: i64,
    %res.offset: !llvm.ptr, %res.gs: !llvm.ptr,
    %res.ls: !llvm.ptr, %res.nws: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c512 = llvm.mlir.constant(512 : i64) : i64
  %range = llvm.alloca %c1 x !llvm.struct<"sycl::_V1::range", opaque>
      : (i32) -> !llvm.ptr
  %offset = llvm.alloca %c1 x !llvm.struct<"sycl::_V1::id", opaque>
      : (i32) -> !llvm.ptr
  sycl.host.constructor(%range, %c512) {type=!sycl_range_1_}
      : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%offset, %offset.0) {type=!sycl_id_1_}
      : (!llvm.ptr, i64) -> ()
  sycl.host.handler.set_nd_range %handler -> range %range, offset %offset
      : !llvm.ptr, !llvm.ptr, !llvm.ptr
  sycl.host.schedule_kernel %handler -> @kernels::@k1[range %range, offset %offset](
      %res.offset, %res.gs, %res.ls, %res.nws)
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// COM: Check we can detect both range and offset are constant

llvm.func internal @foo_constant_range_offset(
    %handler: !llvm.ptr,
    %res.offset.0: !llvm.ptr, %res.offset.1: !llvm.ptr,
    %res.gs.0: !llvm.ptr, %res.gs.1: !llvm.ptr,
    %res.ls.0: !llvm.ptr, %res.ls.1: !llvm.ptr,
    %res.nws.0: !llvm.ptr, %res.nws.1: !llvm.ptr) {
  %c1_i32 = llvm.mlir.constant(1 : i32) : i32
  %c1_i64 = llvm.mlir.constant(1 : i64) : i64
  %c2 = llvm.mlir.constant(1 : i64) : i64
  %c512 = llvm.mlir.constant(512 : i64) : i64
  %c1024 = llvm.mlir.constant(512 : i64) : i64
  %range = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::range.1", opaque>
      : (i32) -> !llvm.ptr
  %offset = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::id.1", opaque>
      : (i32) -> !llvm.ptr
  sycl.host.constructor(%range, %c512, %c1024) {type=!sycl_range_2_}
      : (!llvm.ptr, i64, i64) -> ()
  sycl.host.constructor(%offset, %c1_i64, %c2) {type=!sycl_id_2_}
      : (!llvm.ptr, i64, i64) -> ()
  sycl.host.handler.set_nd_range %handler -> range %range, offset %offset
      : !llvm.ptr, !llvm.ptr, !llvm.ptr
  sycl.host.schedule_kernel %handler -> @kernels::@k2[range %range, offset %offset](
      %res.offset.0, %res.offset.1, %res.gs.0, %res.gs.1,
      %res.ls.0, %res.ls.1, %res.nws.0, %res.nws.1)
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
         !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// COM: Check we can detect default offset (nd-range)

llvm.func internal @foo_default_offset_ndr(
    %handler: !llvm.ptr,
    %i: i64,
    %res.offset.0: !llvm.ptr, %res.offset.1: !llvm.ptr, %res.offset.2: !llvm.ptr,
    %res.gs.0: !llvm.ptr, %res.gs.1: !llvm.ptr,  %res.gs.2: !llvm.ptr,
    %res.ls.0: !llvm.ptr, %res.ls.1: !llvm.ptr, %res.ls.2: !llvm.ptr,
    %res.nws.0: !llvm.ptr, %res.nws.1: !llvm.ptr, %res.nws.2: !llvm.ptr) {
  %c1_i32 = llvm.mlir.constant(1 : i32) : i32
  %c1_i64 = llvm.mlir.constant(1 : i64) : i64
  %c2 = llvm.mlir.constant(1 : i64) : i64
  %c512 = llvm.mlir.constant(512 : i64) : i64
  %c1024 = llvm.mlir.constant(512 : i64) : i64
  %global_size = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::range.2", opaque>
      : (i32) -> !llvm.ptr
  %local_size = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::range.2", opaque>
      : (i32) -> !llvm.ptr
  %nd_range = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::nd_range.2", opaque>
      : (i32) -> !llvm.ptr
  sycl.host.constructor(%global_size, %i, %i, %i) {type=!sycl_range_3_}
      : (!llvm.ptr, i64, i64, i64) -> ()
  sycl.host.constructor(%local_size, %i, %i, %i) {type=!sycl_range_3_}
      : (!llvm.ptr, i64, i64, i64) -> ()
  sycl.host.constructor(%nd_range, %global_size, %local_size)
      {type=!sycl_nd_range_3_}
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.handler.set_nd_range %handler -> nd_range %nd_range : !llvm.ptr, !llvm.ptr
  sycl.host.schedule_kernel %handler -> @kernels::@k3[nd_range %nd_range](
      %res.offset.0, %res.offset.1, %res.offset.2, %res.gs.0,
      %res.gs.1, %res.gs.2, %res.ls.0, %res.ls.1,
      %res.ls.2, %res.nws.0, %res.nws.1, %res.nws.2)
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
         !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
         !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// COM: Check we can detect constant offset (nd-range)

llvm.func internal @foo_constant_constant_offset_ndr(
    %handler: !llvm.ptr,
    %i: i64,
    %res.offset.0: !llvm.ptr, %res.offset.1: !llvm.ptr, %res.offset.2: !llvm.ptr,
    %res.gs.0: !llvm.ptr, %res.gs.1: !llvm.ptr,  %res.gs.2: !llvm.ptr,
    %res.ls.0: !llvm.ptr, %res.ls.1: !llvm.ptr, %res.ls.2: !llvm.ptr,
    %res.nws.0: !llvm.ptr, %res.nws.1: !llvm.ptr, %res.nws.2: !llvm.ptr) {
  %c1_i32 = llvm.mlir.constant(1 : i32) : i32
  %c1_i64 = llvm.mlir.constant(1 : i64) : i64
  %c2 = llvm.mlir.constant(1 : i64) : i64
  %c512 = llvm.mlir.constant(512 : i64) : i64
  %c1024 = llvm.mlir.constant(512 : i64) : i64
  %global_size = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::range.2", opaque>
      : (i32) -> !llvm.ptr
  %local_size = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::range.2", opaque>
      : (i32) -> !llvm.ptr
  %offset = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::id.2", opaque>
      : (i32) -> !llvm.ptr
  %nd_range = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::nd_range.2", opaque>
      : (i32) -> !llvm.ptr
  sycl.host.constructor(%global_size, %i, %i, %i) {type=!sycl_range_3_}
      : (!llvm.ptr, i64, i64, i64) -> ()
  sycl.host.constructor(%local_size, %i, %i, %i) {type=!sycl_range_3_}
      : (!llvm.ptr, i64, i64, i64) -> ()
  sycl.host.constructor(%offset, %c1_i64, %c1_i64, %c1_i64) {type=!sycl_id_3_}
      : (!llvm.ptr, i64, i64, i64) -> ()
  sycl.host.constructor(%nd_range, %global_size, %local_size, %offset)
      {type=!sycl_nd_range_3_}
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.handler.set_nd_range %handler -> nd_range %nd_range : !llvm.ptr, !llvm.ptr
  sycl.host.schedule_kernel %handler -> @kernels::@k4[nd_range %nd_range](
      %res.offset.0, %res.offset.1, %res.offset.2, %res.gs.0,
      %res.gs.1, %res.gs.2, %res.ls.0, %res.ls.1,
      %res.ls.2, %res.nws.0, %res.nws.1, %res.nws.2)
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
         !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
         !llvm.ptr, !llvm.ptr, !llvm.ptr,
         !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// COM: Check we can detect constant global size (nd-range)

llvm.func internal @foo_constant_constant_global_size(
    %handler: !llvm.ptr,
    %i: i64,
    %res.offset.0: !llvm.ptr, %res.offset.1: !llvm.ptr, %res.offset.2: !llvm.ptr,
    %res.gs.0: !llvm.ptr, %res.gs.1: !llvm.ptr,  %res.gs.2: !llvm.ptr,
    %res.ls.0: !llvm.ptr, %res.ls.1: !llvm.ptr, %res.ls.2: !llvm.ptr,
    %res.nws.0: !llvm.ptr, %res.nws.1: !llvm.ptr, %res.nws.2: !llvm.ptr) {
  %c1_i32 = llvm.mlir.constant(1 : i32) : i32
  %c1_i64 = llvm.mlir.constant(1 : i64) : i64
  %c2 = llvm.mlir.constant(1 : i64) : i64
  %c512 = llvm.mlir.constant(512 : i64) : i64
  %c1024 = llvm.mlir.constant(512 : i64) : i64
  %global_size = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::range.2", opaque>
      : (i32) -> !llvm.ptr
  %local_size = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::range.2", opaque>
      : (i32) -> !llvm.ptr
  %offset = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::id.2", opaque>
      : (i32) -> !llvm.ptr
  %nd_range = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::nd_range.2", opaque>
      : (i32) -> !llvm.ptr
  sycl.host.constructor(%global_size, %c512, %c1024, %c2) {type=!sycl_range_3_}
      : (!llvm.ptr, i64, i64, i64) -> ()
  sycl.host.constructor(%local_size, %i, %i, %i) {type=!sycl_range_3_}
      : (!llvm.ptr, i64, i64, i64) -> ()
  sycl.host.constructor(%offset, %i, %i, %i) {type=!sycl_id_3_}
      : (!llvm.ptr, i64, i64, i64) -> ()
  sycl.host.constructor(%nd_range, %global_size, %local_size, %offset)
      {type=!sycl_nd_range_3_}
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.handler.set_nd_range %handler -> nd_range %nd_range : !llvm.ptr, !llvm.ptr
  sycl.host.schedule_kernel %handler -> @kernels::@k5[nd_range %nd_range](
      %res.offset.0, %res.offset.1, %res.offset.2, %res.gs.0,
      %res.gs.1, %res.gs.2, %res.ls.0, %res.ls.1,
      %res.ls.2, %res.nws.0, %res.nws.1, %res.nws.2)
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
         !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
         !llvm.ptr, !llvm.ptr, !llvm.ptr,
         !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// COM: Check we can detect constant local size (nd-range)

llvm.func internal @foo_constant_constant_local_size(
    %handler: !llvm.ptr,
    %i: i64,
    %res.offset.0: !llvm.ptr, %res.offset.1: !llvm.ptr, %res.offset.2: !llvm.ptr,
    %res.gs.0: !llvm.ptr, %res.gs.1: !llvm.ptr,  %res.gs.2: !llvm.ptr,
    %res.ls.0: !llvm.ptr, %res.ls.1: !llvm.ptr, %res.ls.2: !llvm.ptr,
    %res.nws.0: !llvm.ptr, %res.nws.1: !llvm.ptr, %res.nws.2: !llvm.ptr) {
  %c1_i32 = llvm.mlir.constant(1 : i32) : i32
  %c1_i64 = llvm.mlir.constant(1 : i64) : i64
  %c2 = llvm.mlir.constant(1 : i64) : i64
  %c512 = llvm.mlir.constant(512 : i64) : i64
  %c1024 = llvm.mlir.constant(512 : i64) : i64
  %global_size = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::range.2", opaque>
      : (i32) -> !llvm.ptr
  %local_size = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::range.2", opaque>
      : (i32) -> !llvm.ptr
  %offset = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::id.2", opaque>
      : (i32) -> !llvm.ptr
  %nd_range = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::nd_range.2", opaque>
      : (i32) -> !llvm.ptr
  sycl.host.constructor(%global_size, %i, %i, %i) {type=!sycl_range_3_}
      : (!llvm.ptr, i64, i64, i64) -> ()
  sycl.host.constructor(%local_size, %c2, %c512, %c1_i64) {type=!sycl_range_3_}
      : (!llvm.ptr, i64, i64, i64) -> ()
  sycl.host.constructor(%offset, %i, %i, %i) {type=!sycl_id_3_}
      : (!llvm.ptr, i64, i64, i64) -> ()
  sycl.host.constructor(%nd_range, %global_size, %local_size, %offset)
      {type=!sycl_nd_range_3_}
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.handler.set_nd_range %handler -> nd_range %nd_range : !llvm.ptr, !llvm.ptr
  sycl.host.schedule_kernel %handler -> @kernels::@k6[nd_range %nd_range](
      %res.offset.0, %res.offset.1, %res.offset.2, %res.gs.0,
      %res.gs.1, %res.gs.2, %res.ls.0, %res.ls.1,
      %res.ls.2, %res.nws.0, %res.nws.1, %res.nws.2)
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
         !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
         !llvm.ptr, !llvm.ptr, !llvm.ptr,
         !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// COM: Check we can detect constant global and local size (nd-range)

llvm.func internal @foo_constant_constant_global_local_size(
    %handler: !llvm.ptr,
    %i: i64,
    %res.offset.0: !llvm.ptr, %res.offset.1: !llvm.ptr, %res.offset.2: !llvm.ptr,
    %res.gs.0: !llvm.ptr, %res.gs.1: !llvm.ptr,  %res.gs.2: !llvm.ptr,
    %res.ls.0: !llvm.ptr, %res.ls.1: !llvm.ptr, %res.ls.2: !llvm.ptr,
    %res.nws.0: !llvm.ptr, %res.nws.1: !llvm.ptr, %res.nws.2: !llvm.ptr) {
  %c1_i32 = llvm.mlir.constant(1 : i32) : i32
  %c1_i64 = llvm.mlir.constant(1 : i64) : i64
  %c2 = llvm.mlir.constant(1 : i64) : i64
  %c512 = llvm.mlir.constant(512 : i64) : i64
  %c1024 = llvm.mlir.constant(512 : i64) : i64
  %global_size = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::range.2", opaque>
      : (i32) -> !llvm.ptr
  %local_size = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::range.2", opaque>
      : (i32) -> !llvm.ptr
  %offset = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::id.2", opaque>
      : (i32) -> !llvm.ptr
  %nd_range = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::nd_range.2", opaque>
      : (i32) -> !llvm.ptr
  sycl.host.constructor(%global_size, %c512, %c1024, %c2) {type=!sycl_range_3_}
      : (!llvm.ptr, i64, i64, i64) -> ()
  sycl.host.constructor(%local_size, %c2, %c512, %c1_i64) {type=!sycl_range_3_}
      : (!llvm.ptr, i64, i64, i64) -> ()
  sycl.host.constructor(%offset, %i, %i, %i) {type=!sycl_id_3_}
      : (!llvm.ptr, i64, i64, i64) -> ()
  sycl.host.constructor(%nd_range, %global_size, %local_size, %offset)
      {type=!sycl_nd_range_3_}
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.handler.set_nd_range %handler -> nd_range %nd_range : !llvm.ptr, !llvm.ptr
  sycl.host.schedule_kernel %handler -> @kernels::@k7[nd_range %nd_range](
      %res.offset.0, %res.offset.1, %res.offset.2, %res.gs.0,
      %res.gs.1, %res.gs.2, %res.ls.0, %res.ls.1,
      %res.ls.2, %res.nws.0, %res.nws.1, %res.nws.2)
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
         !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
         !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// -----

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
!sycl_accessor_1_i32_w_gb = !sycl.accessor<[1, i32, write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_accessor_host = !sycl.accessor<[1, i32, write, global_buffer], (!llvm.void)>
!sycl_accessor_host_range = !sycl.accessor<[1, i32, write, global_buffer], (!sycl_range_1_)>
!sycl_accessor_host_offset = !sycl.accessor<[1, i32, write, global_buffer], (!sycl_range_1_, !sycl_id_1_)>

gpu.module @kernels0 {
  func.func private @init(%acc: memref<1x!sycl_accessor_1_i32_w_gb>,
                          %ptr: memref<?xi32, 1>,
                          %accRange: memref<?x!sycl_range_1_>,
                          %memRange: memref<?x!sycl_range_1_>,
                          %offset: memref<?x!sycl_id_1_>)

// COM: Constant access range and offset

// CHECK-LABEL:     gpu.func @k0(
// CHECK-SAME:                   %[[VAL_0:.*]]: memref<?xi32, 1>, %[[VAL_1:.*]]: memref<?x!sycl_range_1_>, %[[VAL_2:.*]]: memref<?x!sycl_range_1_>, %[[VAL_3:.*]]: memref<?x!sycl_id_1_>) kernel {
// CHECK-NEXT:        %[[VAL_4:.*]] = arith.constant 512 : index
// CHECK-NEXT:        %[[VAL_5:.*]] = sycl.range.constructor(%[[VAL_4]]) : (index) -> memref<1x!sycl_range_1_>
// CHECK-NEXT:        %[[VAL_6:.*]] = memref.cast %[[VAL_5]] : memref<1x!sycl_range_1_> to memref<?x!sycl_range_1_>
// CHECK-NEXT:        %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_8:.*]] = sycl.id.constructor(%[[VAL_7]]) : (index) -> memref<1x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_9:.*]] = memref.cast %[[VAL_8]] : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_10:.*]] = memref.alloca() : memref<1x!sycl_accessor_1_i32_w_gb2>
// CHECK-NEXT:        func.call @init(%[[VAL_10]], %[[VAL_0]], %[[VAL_6]], %[[VAL_2]], %[[VAL_9]]) : (memref<1x!sycl_accessor_1_i32_w_gb2>, memref<?xi32, 1>, memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>, memref<?x!sycl_id_1_>) -> ()
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }
  gpu.func @k0(%ptr: memref<?xi32, 1>,
               %accRange: memref<?x!sycl_range_1_>,
               %memRange: memref<?x!sycl_range_1_>,
               %offset: memref<?x!sycl_id_1_>) kernel {
    %acc = memref.alloca() : memref<1x!sycl_accessor_1_i32_w_gb>
    func.call @init(%acc, %ptr, %accRange, %memRange, %offset)
        : (memref<1x!sycl_accessor_1_i32_w_gb>, memref<?xi32, 1>,
           memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>,
           memref<?x!sycl_id_1_>) -> ()
    gpu.return
  }
}

llvm.func internal @constant_offset_and_range(
    %lambda: !llvm.ptr, %buffer: !llvm.ptr, %handler: !llvm.ptr) {
  %c1_i32 = llvm.mlir.constant(1 : i32) : i32
  %c1_i64 = llvm.mlir.constant(1 : i64) : i64
  %c512 = llvm.mlir.constant(512 : i64) : i64
  %range = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::range", opaque>
      : (i32) -> !llvm.ptr
  %offset = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::id", opaque>
      : (i32) -> !llvm.ptr
  %acc = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::accessor", opaque>
      : (i32) -> !llvm.ptr
  sycl.host.constructor(%range, %c512) {type=!sycl_range_1_}
      : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%offset, %c1_i64) {type=!sycl_id_1_}
      : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%acc, %buffer, %handler, %range, %offset)
      {type=!sycl_accessor_host_offset}
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.set_captured %lambda[0] = %acc : !llvm.ptr, !llvm.ptr (!sycl_accessor_host_offset)
  sycl.host.schedule_kernel %handler -> @kernels0::@k0(%acc: !sycl_accessor_host_offset)
      : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// COM: No access range needed, unknown memory range and default offset.

gpu.module @kernels1 {
  func.func private @init(%acc: memref<1x!sycl_accessor_1_i32_w_gb>,
                          %ptr: memref<?xi32, 1>,
                          %accRange: memref<?x!sycl_range_1_>,
                          %memRange: memref<?x!sycl_range_1_>,
                          %offset: memref<?x!sycl_id_1_>)

// CHECK-LABEL:     gpu.func @k1(
// CHECK-SAME:                   %[[VAL_11:.*]]: memref<?xi32, 1>, %[[VAL_12:.*]]: memref<?x!sycl_range_1_>, %[[VAL_13:.*]]: memref<?x!sycl_range_1_>, %[[VAL_14:.*]]: memref<?x!sycl_id_1_>) kernel {
// CHECK-NEXT:        %[[VAL_15:.*]] = sycl.id.constructor() : () -> memref<1x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_16:.*]] = memref.cast %[[VAL_15]] : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_17:.*]] = memref.alloca() : memref<1x!sycl_accessor_1_i32_w_gb2>
// CHECK-NEXT:        func.call @init(%[[VAL_17]], %[[VAL_11]], %[[VAL_13]], %[[VAL_13]], %[[VAL_16]]) : (memref<1x!sycl_accessor_1_i32_w_gb2>, memref<?xi32, 1>, memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>, memref<?x!sycl_id_1_>) -> ()
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }
  gpu.func @k1(%ptr: memref<?xi32, 1>,
               %accRange: memref<?x!sycl_range_1_>,
               %memRange: memref<?x!sycl_range_1_>,
               %offset: memref<?x!sycl_id_1_>) kernel {
    %acc = memref.alloca() : memref<1x!sycl_accessor_1_i32_w_gb>
    func.call @init(%acc, %ptr, %accRange, %memRange, %offset)
        : (memref<1x!sycl_accessor_1_i32_w_gb>, memref<?xi32, 1>,
           memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>,
           memref<?x!sycl_id_1_>) -> ()
    gpu.return
  }
}

llvm.func internal @unknown_buffer_range_default_offset(
    %lambda: !llvm.ptr, %ptr: !llvm.ptr, %handler: !llvm.ptr, %range: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %buffer = llvm.alloca %c1 x !llvm.struct<"sycl::_V1::buffer", opaque>
      : (i32) -> !llvm.ptr
  %acc = llvm.alloca %c1 x !llvm.struct<"sycl::_V1::accessor", opaque>
      : (i32) -> !llvm.ptr
  sycl.host.constructor(%buffer, %ptr, %range)
      {type=!sycl.buffer<[1, i32]>}
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%acc, %buffer, %handler)
      {type=!sycl_accessor_host}
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.set_captured %lambda[0] = %acc : !llvm.ptr, !llvm.ptr (!sycl_accessor_host)
  sycl.host.schedule_kernel %handler -> @kernels1::@k1(%acc: !sycl_accessor_host)
      : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// COM: No access range needed, constant memory range and default offset.

gpu.module @kernels2 {
  func.func private @init(%acc: memref<1x!sycl_accessor_1_i32_w_gb>,
                          %ptr: memref<?xi32, 1>,
                          %accRange: memref<?x!sycl_range_1_>,
                          %memRange: memref<?x!sycl_range_1_>,
                          %offset: memref<?x!sycl_id_1_>)

// CHECK-LABEL:     gpu.func @k2(
// CHECK-SAME:                   %[[VAL_18:.*]]: memref<?xi32, 1>, %[[VAL_19:.*]]: memref<?x!sycl_range_1_>, %[[VAL_20:.*]]: memref<?x!sycl_range_1_>, %[[VAL_21:.*]]: memref<?x!sycl_id_1_>) kernel {
// CHECK-NEXT:        %[[VAL_22:.*]] = arith.constant 512 : index
// CHECK-NEXT:        %[[VAL_23:.*]] = sycl.range.constructor(%[[VAL_22]]) : (index) -> memref<1x!sycl_range_1_>
// CHECK-NEXT:        %[[VAL_24:.*]] = memref.cast %[[VAL_23]] : memref<1x!sycl_range_1_> to memref<?x!sycl_range_1_>
// CHECK-NEXT:        %[[VAL_25:.*]] = sycl.id.constructor() : () -> memref<1x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_26:.*]] = memref.cast %[[VAL_25]] : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_27:.*]] = memref.alloca() : memref<1x!sycl_accessor_1_i32_w_gb2>
// CHECK-NEXT:        func.call @init(%[[VAL_27]], %[[VAL_18]], %[[VAL_24]], %[[VAL_24]], %[[VAL_26]]) : (memref<1x!sycl_accessor_1_i32_w_gb2>, memref<?xi32, 1>, memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>, memref<?x!sycl_id_1_>) -> ()
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }
  gpu.func @k2(%ptr: memref<?xi32, 1>,
               %accRange: memref<?x!sycl_range_1_>,
               %memRange: memref<?x!sycl_range_1_>,
               %offset: memref<?x!sycl_id_1_>) kernel {
    %acc = memref.alloca() : memref<1x!sycl_accessor_1_i32_w_gb>
    func.call @init(%acc, %ptr, %accRange, %memRange, %offset)
        : (memref<1x!sycl_accessor_1_i32_w_gb>, memref<?xi32, 1>,
           memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>,
           memref<?x!sycl_id_1_>) -> ()
    gpu.return
  }
}

llvm.func internal @known_buffer_range_default_offset(
    %lambda: !llvm.ptr, %ptr: !llvm.ptr, %handler: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c512 = llvm.mlir.constant(512 : i64) : i64
  %range = llvm.alloca %c1 x !llvm.struct<"sycl::_V1::range", opaque>
      : (i32) -> !llvm.ptr
  %buffer = llvm.alloca %c1 x !llvm.struct<"sycl::_V1::buffer", opaque>
      : (i32) -> !llvm.ptr
  %acc = llvm.alloca %c1 x !llvm.struct<"sycl::_V1::accessor", opaque>
      : (i32) -> !llvm.ptr
  sycl.host.constructor(%range, %c512) {type=!sycl_range_1_}
      : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%buffer, %ptr, %range)
      {type=!sycl.buffer<[1, i32]>}
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%acc, %buffer, %handler)
      {type=!sycl_accessor_host}
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.set_captured %lambda[0] = %acc : !llvm.ptr, !llvm.ptr (!sycl_accessor_host)
  sycl.host.schedule_kernel %handler -> @kernels2::@k2(%acc: !sycl_accessor_host)
      : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// COM: Not enough info

gpu.module @kernels3 {
  func.func private @init(%acc: memref<1x!sycl_accessor_1_i32_w_gb>,
                          %ptr: memref<?xi32, 1>,
                          %accRange: memref<?x!sycl_range_1_>,
                          %memRange: memref<?x!sycl_range_1_>,
                          %offset: memref<?x!sycl_id_1_>)

// CHECK-LABEL:     gpu.func @k3(
// CHECK-SAME:                   %[[VAL_28:.*]]: memref<?xi32, 1>, %[[VAL_29:.*]]: memref<?x!sycl_range_1_>, %[[VAL_30:.*]]: memref<?x!sycl_range_1_>, %[[VAL_31:.*]]: memref<?x!sycl_id_1_>) kernel {
// CHECK-NEXT:        %[[VAL_32:.*]] = memref.alloca() : memref<1x!sycl_accessor_1_i32_w_gb2>
// CHECK-NEXT:        func.call @init(%[[VAL_32]], %[[VAL_28]], %[[VAL_29]], %[[VAL_30]], %[[VAL_31]]) : (memref<1x!sycl_accessor_1_i32_w_gb2>, memref<?xi32, 1>, memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>, memref<?x!sycl_id_1_>) -> ()
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }
  gpu.func @k3(%ptr: memref<?xi32, 1>,
               %accRange: memref<?x!sycl_range_1_>,
               %memRange: memref<?x!sycl_range_1_>,
               %offset: memref<?x!sycl_id_1_>) kernel {
    %acc = memref.alloca() : memref<1x!sycl_accessor_1_i32_w_gb>
    func.call @init(%acc, %ptr, %accRange, %memRange, %offset)
        : (memref<1x!sycl_accessor_1_i32_w_gb>, memref<?xi32, 1>,
           memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>,
           memref<?x!sycl_id_1_>) -> ()
    gpu.return
  }
}

llvm.func internal @unknown_offset_and_range(
    %lambda: !llvm.ptr, %buffer: !llvm.ptr, %handler: !llvm.ptr,
    %range: !llvm.ptr, %offset: !llvm.ptr) {
  %c1_i32 = llvm.mlir.constant(1 : i32) : i32
  %c1_i64 = llvm.mlir.constant(1 : i64) : i64
  %c512 = llvm.mlir.constant(512 : i64) : i64
  %acc = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::accessor", opaque>
      : (i32) -> !llvm.ptr
  sycl.host.constructor(%acc, %buffer, %handler, %range, %offset)
      {type=!sycl_accessor_host_offset}
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.set_captured %lambda[0] = %acc : !llvm.ptr, !llvm.ptr (!sycl_accessor_host_offset)
  sycl.host.schedule_kernel %handler -> @kernels3::@k3(%acc: !sycl_accessor_host_offset)
      : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// -----

// COM: Check we can propagate '@constant_array'

gpu.module @kernels {
  // CHECK-LABEL:      llvm.mlir.global private unnamed_addr constant @constant_array(dense<[1.300000e+02, 1.200000e+02, 7.800000e+01, 0.000000e+00, -7.800000e+01, -1.200000e+02, -1.300000e+02, 1.800000e+02, 1.950000e+02, 1.560000e+02, 0.000000e+00, -1.560000e+02, -1.950000e+02, -1.800000e+02, 2.340000e+02, 3.120000e+02, 3.900000e+02, 0.000000e+00, -3.900000e+02, -3.120000e+02, -2.340000e+02, 2.600000e+02, 3.900000e+02, 7.800000e+02, 0.000000e+00, -7.800000e+02, -3.900000e+02, -2.600000e+02, 2.340000e+02, 3.120000e+02, 3.900000e+02, 0.000000e+00, -3.900000e+02, -3.120000e+02, -2.340000e+02, 1.800000e+02, 1.950000e+02, 1.560000e+02, 0.000000e+00, -1.560000e+02, -1.950000e+02, -1.800000e+02, 1.300000e+02, 1.200000e+02, 7.800000e+01, 0.000000e+00, -7.800000e+01, -1.200000e+02, -1.300000e+02]> : tensor<49xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<49 x f32>

// CHECK-LABEL:     gpu.func @k(
// CHECK-SAME:                  %[[VAL_0:.*]]: memref<?xf32, 1>,
// CHECK-SAME:                  %[[VAL_1:.*]]: !llvm.ptr {llvm.align = 4 : i64, llvm.byval = !llvm.struct<(array<49 x f32>)>, llvm.noundef}) kernel {
// CHECK-NEXT:        %[[VAL_2:.*]] = llvm.mlir.addressof @constant_array : !llvm.ptr
// CHECK-NEXT:        %[[VAL_4:.*]] = arith.constant 49 : index
// CHECK-NEXT:        affine.for %[[VAL_5:.*]] = 0 to %[[VAL_4]] {
// CHECK-NEXT:          %[[VAL_6:.*]] = arith.index_cast %[[VAL_5]] : index to i64
// CHECK-NEXT:          %[[VAL_7:.*]] = llvm.getelementptr %[[VAL_2]][0, 0, %[[VAL_6]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(array<49 x f32>)>
// CHECK-NEXT:          %[[VAL_8:.*]] = llvm.load %[[VAL_7]] : !llvm.ptr -> f32
// CHECK-NEXT:          affine.store %[[VAL_8]], %[[VAL_0]]{{\[}}%[[VAL_5]]] : memref<?xf32, 1>
// CHECK-NEXT:        }
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }
  gpu.func @k(%ptr: memref<?xf32, 1>,
              %const_arr: !llvm.ptr
                  {llvm.align = 4 : i64,
                   llvm.byval = !llvm.struct<(array<49 x f32>)>,
                   llvm.noundef})
        kernel {
    %c48 = arith.constant 49 : index
    affine.for %i = 0 to %c48 {
      %i_i32 = arith.index_cast %i : index to i64
      %arr_ptr = llvm.getelementptr %const_arr[0, 0, %i_i32] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(array<49 x f32>)>
      %val = llvm.load %arr_ptr : !llvm.ptr -> f32
      affine.store %val, %ptr[%i] : memref<?xf32, 1>
    }
    gpu.return
  }
}

llvm.mlir.global private unnamed_addr constant @constant_array(dense<[1.300000e+02, 1.200000e+02, 7.800000e+01, 0.000000e+00, -7.800000e+01, -1.200000e+02, -1.300000e+02, 1.800000e+02, 1.950000e+02, 1.560000e+02, 0.000000e+00, -1.560000e+02, -1.950000e+02, -1.800000e+02, 2.340000e+02, 3.120000e+02, 3.900000e+02, 0.000000e+00, -3.900000e+02, -3.120000e+02, -2.340000e+02, 2.600000e+02, 3.900000e+02, 7.800000e+02, 0.000000e+00, -7.800000e+02, -3.900000e+02, -2.600000e+02, 2.340000e+02, 3.120000e+02, 3.900000e+02, 0.000000e+00, -3.900000e+02, -3.120000e+02, -2.340000e+02, 1.800000e+02, 1.950000e+02, 1.560000e+02, 0.000000e+00, -1.560000e+02, -1.950000e+02, -1.800000e+02, 1.300000e+02, 1.200000e+02, 7.800000e+01, 0.000000e+00, -7.800000e+01, -1.200000e+02, -1.300000e+02]> : tensor<49xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<49 x f32>

llvm.func internal @propagate_array(%ptr: !llvm.ptr, %handler: !llvm.ptr) {
  %arr = llvm.mlir.addressof @constant_array : !llvm.ptr
  sycl.host.schedule_kernel %handler -> @kernels::@k(%ptr, %arr)
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// -----

// COM: These tests check that (nd-)range, offset and captured accessors can be
// COM: modified after 'sycl.host.handler.set_nd_range' (for nd-range info) or
// COM: 'sycl.host.set_captured' for accessors.

!sycl_range_1_ = !sycl.range<[1], (memref<1xi64>)>
!sycl_id_1_ = !sycl.id<[1], (memref<1xi64>)>
!sycl_nd_range_1_ = !sycl.nd_range<[1], (!sycl_range_1_, !sycl_range_1_, !sycl_id_1_)>
!sycl_accessor_1_i32_w_gb = !sycl.accessor<[1, i32, write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_accessor_host = !sycl.accessor<[1, i32, write, global_buffer], (!llvm.void)>

gpu.module @kernels {

// COM: Check we propagate range and offset information

// CHECK-LABEL:     gpu.func @k0(
// CHECK-SAME:                   %[[VAL_0:.*]]: memref<?xi64, 1>, %[[VAL_1:.*]]: memref<?xi64, 1>) kernel {
// CHECK-NEXT:        %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_3:.*]] = arith.constant 512 : index
// CHECK-NEXT:        %[[VAL_4:.*]] = sycl.range.constructor(%[[VAL_3]]) : (index) -> memref<1x!sycl_range_1_>
// CHECK-NEXT:        %[[VAL_5:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<1x!sycl_range_1_>
// CHECK-NEXT:        %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_8:.*]] = sycl.id.constructor(%[[VAL_7]]) : (index) -> memref<1x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_9:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_6]]] : memref<1x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_10:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[VAL_11:.*]] = memref.alloca() : memref<!sycl_range_1_>
// CHECK-NEXT:        %[[VAL_12:.*]] = memref.alloca() : memref<!sycl_id_1_>
// CHECK-NEXT:        affine.store %[[VAL_5]], %[[VAL_11]][] : memref<!sycl_range_1_>
// CHECK-NEXT:        affine.store %[[VAL_9]], %[[VAL_12]][] : memref<!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_13:.*]] = sycl.range.get %[[VAL_11]]{{\[}}%[[VAL_10]]] : (memref<!sycl_range_1_>, i32) -> i64
// CHECK-NEXT:        affine.store %[[VAL_13]], %[[VAL_0]][0] : memref<?xi64, 1>
// CHECK-NEXT:        %[[VAL_14:.*]] = sycl.id.get %[[VAL_12]][] : (memref<!sycl_id_1_>) -> i64
// CHECK-NEXT:        affine.store %[[VAL_14]], %[[VAL_1]][0] : memref<?xi64, 1>
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }

  gpu.func @k0(%ptr0: memref<?xi64, 1>, %ptr1: memref<?xi64, 1>) kernel {
    %c0 = arith.constant 0 : i32
    %range = memref.alloca() : memref<!sycl_range_1_>
    %id = memref.alloca() : memref<!sycl_id_1_>
    %gs = sycl.num_work_items : !sycl_range_1_
    affine.store %gs, %range[] : memref<!sycl_range_1_>
    %off = sycl.global_offset : !sycl_id_1_
    affine.store %off, %id[] : memref<!sycl_id_1_>
    %res0 = sycl.range.get %range[%c0] : (memref<!sycl_range_1_>, i32) -> i64
    affine.store %res0, %ptr0[0] : memref<?xi64, 1>
    %res1 = sycl.id.get %id[] : (memref<!sycl_id_1_>) -> i64
    affine.store %res1, %ptr1[0] : memref<?xi64, 1>
    gpu.return
  }

// COM: Check we propagate nd_range information

// CHECK-LABEL:     gpu.func @k1(
// CHECK-SAME:                   %[[VAL_15:.*]]: memref<?xi64, 1>, %[[VAL_16:.*]]: memref<?xi64, 1>) kernel {
// CHECK-NEXT:        %[[VAL_17:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_18:.*]] = arith.constant 512 : index
// CHECK-NEXT:        %[[VAL_19:.*]] = sycl.range.constructor(%[[VAL_18]]) : (index) -> memref<1x!sycl_range_1_>
// CHECK-NEXT:        %[[VAL_20:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_17]]] : memref<1x!sycl_range_1_>
// CHECK-NEXT:        %[[VAL_21:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_22:.*]] = arith.constant 1 : index
// CHECK-NEXT:        %[[VAL_23:.*]] = sycl.id.constructor(%[[VAL_22]]) : (index) -> memref<1x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_24:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_21]]] : memref<1x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_25:.*]] = arith.constant 0 : i32
// CHECK-NEXT:        %[[VAL_26:.*]] = memref.alloca() : memref<!sycl_range_1_>
// CHECK-NEXT:        %[[VAL_27:.*]] = memref.alloca() : memref<!sycl_id_1_>
// CHECK-NEXT:        affine.store %[[VAL_20]], %[[VAL_26]][] : memref<!sycl_range_1_>
// CHECK-NEXT:        affine.store %[[VAL_24]], %[[VAL_27]][] : memref<!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_28:.*]] = sycl.range.get %[[VAL_26]]{{\[}}%[[VAL_25]]] : (memref<!sycl_range_1_>, i32) -> i64
// CHECK-NEXT:        affine.store %[[VAL_28]], %[[VAL_15]][0] : memref<?xi64, 1>
// CHECK-NEXT:        %[[VAL_29:.*]] = sycl.id.get %[[VAL_27]][] : (memref<!sycl_id_1_>) -> i64
// CHECK-NEXT:        affine.store %[[VAL_29]], %[[VAL_16]][0] : memref<?xi64, 1>
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }

  gpu.func @k1(%ptr0: memref<?xi64, 1>, %ptr1: memref<?xi64, 1>) kernel {
    %c0 = arith.constant 0 : i32
    %range = memref.alloca() : memref<!sycl_range_1_>
    %id = memref.alloca() : memref<!sycl_id_1_>
    %gs = sycl.num_work_items : !sycl_range_1_
    affine.store %gs, %range[] : memref<!sycl_range_1_>
    %off = sycl.global_offset : !sycl_id_1_
    affine.store %off, %id[] : memref<!sycl_id_1_>
    %res0 = sycl.range.get %range[%c0] : (memref<!sycl_range_1_>, i32) -> i64
    affine.store %res0, %ptr0[0] : memref<?xi64, 1>
    %res1 = sycl.id.get %id[] : (memref<!sycl_id_1_>) -> i64
    affine.store %res1, %ptr1[0] : memref<?xi64, 1>
    gpu.return
  }

  func.func private @__init(
      memref<1x!sycl_accessor_1_i32_w_gb>, memref<?xi32, 1>,
      memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>,
      memref<?x!sycl_id_1_>)

// COM: Check we do not use access range (arg #1) and offset (arg #3)

// CHECK-LABEL:     gpu.func @k2(
// CHECK-SAME:                   %[[VAL_30:.*]]: memref<?xi32, 1>, %[[VAL_31:.*]]: memref<?x!sycl_range_1_>, %[[VAL_32:.*]]: memref<?x!sycl_range_1_>, %[[VAL_33:.*]]: memref<?x!sycl_id_1_>) kernel {
// CHECK-NEXT:        %[[VAL_34:.*]] = sycl.id.constructor() : () -> memref<1x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_35:.*]] = memref.cast %[[VAL_34]] : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
// CHECK-NEXT:        %[[VAL_36:.*]] = memref.alloca() : memref<1x!sycl_accessor_1_i32_w_gb1>
// CHECK-NEXT:        func.call @__init(%[[VAL_36]], %[[VAL_30]], %[[VAL_32]], %[[VAL_32]], %[[VAL_35]]) : (memref<1x!sycl_accessor_1_i32_w_gb1>, memref<?xi32, 1>, memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>, memref<?x!sycl_id_1_>) -> ()
// CHECK-NEXT:        gpu.return
// CHECK-NEXT:      }

  gpu.func @k2(%ptr: memref<?xi32, 1>,
               %accRange: memref<?x!sycl_range_1_>,
               %memRange: memref<?x!sycl_range_1_>,
               %offset: memref<?x!sycl_id_1_>) kernel {
    %acc = memref.alloca() : memref<1x!sycl_accessor_1_i32_w_gb>
    func.call @__init(%acc, %ptr, %accRange, %memRange, %offset)
        : (memref<1x!sycl_accessor_1_i32_w_gb>, memref<?xi32, 1>,
           memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>,
           memref<?x!sycl_id_1_>) -> ()
    gpu.return
  }
}

llvm.func internal @constant_range_offset(
    %handler: !llvm.ptr, %res.gs: !llvm.ptr, %res.offset: !llvm.ptr) {
  %c1_i32 = llvm.mlir.constant(1 : i32) : i32
  %c2 = llvm.mlir.constant(1 : i64) : i64
  %c1024 = llvm.mlir.constant(512 : i64) : i64
  %range = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::range.1", opaque>
      : (i32) -> !llvm.ptr
  %offset = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::id.1", opaque>
      : (i32) -> !llvm.ptr
  sycl.host.constructor(%range, %c1024) {type=!sycl_range_1_}
      : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%offset, %c2) {type=!sycl_id_1_}
      : (!llvm.ptr, i64) -> ()
  sycl.host.handler.set_nd_range %handler -> range %range, offset %offset
      : !llvm.ptr, !llvm.ptr, !llvm.ptr
  // COM: These stores should be omitted
  llvm.store %c2, %range : i64, !llvm.ptr
  llvm.store %c1024, %offset : i64, !llvm.ptr
  sycl.host.schedule_kernel %handler -> @kernels::@k0[range %range, offset %offset](
      %res.gs, %res.offset)
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

llvm.func internal @constant_nd_range(
    %handler: !llvm.ptr, %res.gs: !llvm.ptr, %res.offset: !llvm.ptr) {
  %c1_i32 = llvm.mlir.constant(1 : i32) : i32
  %c2 = llvm.mlir.constant(1 : i64) : i64
  %c1024 = llvm.mlir.constant(512 : i64) : i64
  %gs = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::range.1", opaque>
      : (i32) -> !llvm.ptr
  %ls = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::range.1", opaque>
      : (i32) -> !llvm.ptr
  %offset = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::id.1", opaque>
      : (i32) -> !llvm.ptr
  %nd_range = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::nd_range.1", opaque>
      : (i32) -> !llvm.ptr
  sycl.host.constructor(%gs, %c1024) {type=!sycl_range_1_}
      : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%ls, %c2) {type=!sycl_range_1_}
      : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%offset, %c2) {type=!sycl_id_1_}
      : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%nd_range, %gs, %ls, %offset) {type=!sycl_nd_range_1_}
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.handler.set_nd_range %handler -> nd_range %nd_range
      : !llvm.ptr, !llvm.ptr
  // COM: This store should be omitted
  llvm.store %c2, %nd_range : i64, !llvm.ptr
  sycl.host.schedule_kernel %handler -> @kernels::@k1[nd_range %nd_range](
      %res.gs, %res.offset)
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

llvm.func internal @constant_acc(
    %handler: !llvm.ptr, %lambda: !llvm.ptr, %buffer: !llvm.ptr, %range: !llvm.ptr, %ptr: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %accessor = llvm.alloca %c1 x !llvm.struct<"sycl::_V1::accessor.1", opaque>
      : (i32) -> !llvm.ptr
  sycl.host.constructor(%accessor, %buffer, %handler) {type=!sycl_accessor_host}
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.set_captured %lambda[0] = %accessor : !llvm.ptr, !llvm.ptr (!sycl_accessor_host)
  // COM: This store should be omitted
  llvm.store %c1, %accessor : i32, !llvm.ptr
  sycl.host.schedule_kernel %handler -> @kernels::@k2[range %range](%accessor: !sycl_accessor_host)
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// -----

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_local_accessor_1_i64 = !sycl.local_accessor<[1, i64], (memref<?xi64, 1>, memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>, memref<?x!sycl_id_1_>)>
!sycl_local_accessor_host = !sycl.local_accessor<[1, !llvm.void], (!sycl_range_1_)>

gpu.module @kernels0 {
  func.func private @init(%acc: memref<1x!sycl_local_accessor_1_i64>,
                          %ptr: memref<?xi64, 1>,
                          %accRange: memref<?x!sycl_range_1_>,
                          %memRange: memref<?x!sycl_range_1_>,
                          %offset: memref<?x!sycl_id_1_>)

// COM: Check we use zero-initialized range/id as memory range/offset

// CHECK-LABEL:     gpu.func @k(
// CHECK-SAME:                  %[[VAL_0:.*]]: memref<?xi64, 1>, %[[VAL_1:.*]]: memref<?x!sycl_range_1_>, %[[VAL_2:.*]]: memref<?x!sycl_range_1_>, %[[VAL_3:.*]]: memref<?x!sycl_id_1_>) kernel {
// CHECK-NEXT:             %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[VAL_4:.*]] = sycl.range.constructor(%[[C0]]) : (index) -> memref<1x!sycl_range_1_>
// CHECK-NEXT:             %[[VAL_5:.*]] = memref.cast %[[VAL_4]] : memref<1x!sycl_range_1_> to memref<?x!sycl_range_1_>
// CHECK-NEXT:             %[[VAL_6:.*]] = sycl.id.constructor() : () -> memref<1x!sycl_id_1_>
// CHECK-NEXT:             %[[VAL_7:.*]] = memref.cast %[[VAL_6]] : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
// CHECK-NEXT:             %[[VAL_8:.*]] = memref.alloca() : memref<1x!sycl_local_accessor_1_i64_>
// CHECK-NEXT:             func.call @init(%[[VAL_8]], %[[VAL_0]], %[[VAL_1]], %[[VAL_5]], %[[VAL_7]]) : (memref<1x!sycl_local_accessor_1_i64_>, memref<?xi64, 1>, memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>, memref<?x!sycl_id_1_>) -> ()
// CHECK-NEXT:             gpu.return
// CHECK-NEXT:           }
  gpu.func @k(%ptr: memref<?xi64, 1>,
              %accRange: memref<?x!sycl_range_1_>,
              %memRange: memref<?x!sycl_range_1_>,
              %offset: memref<?x!sycl_id_1_>) kernel {
    %acc = memref.alloca() : memref<1x!sycl_local_accessor_1_i64>
    func.call @init(%acc, %ptr, %accRange, %memRange, %offset)
        : (memref<1x!sycl_local_accessor_1_i64>, memref<?xi64, 1>,
           memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>,
           memref<?x!sycl_id_1_>) -> ()
    gpu.return
  }
}

llvm.func internal @unknown_range(
  %lambda: !llvm.ptr, %handler: !llvm.ptr, %range: !llvm.ptr) {
  %c1_i32 = llvm.mlir.constant(1 : i32) : i32
  %acc = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::local_accessor", opaque>
      : (i32) -> !llvm.ptr
  sycl.host.constructor(%acc, %range, %handler)
      {type=!sycl_local_accessor_host}
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.set_captured %lambda[0] = %acc : !llvm.ptr, !llvm.ptr (!sycl_local_accessor_host)
  sycl.host.schedule_kernel %handler -> @kernels0::@k(%acc: !sycl_local_accessor_host)
      : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

gpu.module @kernels1 {
  func.func private @init(%acc: memref<1x!sycl_local_accessor_1_i64>,
                          %ptr: memref<?xi64, 1>,
                          %accRange: memref<?x!sycl_range_1_>,
                          %memRange: memref<?x!sycl_range_1_>,
                          %offset: memref<?x!sycl_id_1_>)

// COM: Check we use 0-initialized range/id as memory range/offset and constant access range

// CHECK-LABEL:     gpu.func @k(
// CHECK-SAME:        %[[VAL_0:.*]]: memref<?xi64, 1>, %[[VAL_1:.*]]: memref<?x!sycl_range_1_>, %[[VAL_2:.*]]: memref<?x!sycl_range_1_>, %[[VAL_3:.*]]: memref<?x!sycl_id_1_>) kernel {
// CHECK-NEXT:             %[[VAL_4:.*]] = arith.constant 512 : index
// CHECK-NEXT:             %[[VAL_5:.*]] = sycl.range.constructor(%[[VAL_4]]) : (index) -> memref<1x!sycl_range_1_>
// CHECK-NEXT:             %[[VAL_6:.*]] = memref.cast %[[VAL_5]] : memref<1x!sycl_range_1_> to memref<?x!sycl_range_1_>
// CHECK-NEXT:             %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[VAL_8:.*]] = sycl.range.constructor(%[[VAL_7]]) : (index) -> memref<1x!sycl_range_1_>
// CHECK-NEXT:             %[[VAL_9:.*]] = memref.cast %[[VAL_8]] : memref<1x!sycl_range_1_> to memref<?x!sycl_range_1_>
// CHECK-NEXT:             %[[VAL_10:.*]] = sycl.id.constructor() : () -> memref<1x!sycl_id_1_>
// CHECK-NEXT:             %[[VAL_11:.*]] = memref.cast %[[VAL_10]] : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
// CHECK-NEXT:             %[[VAL_12:.*]] = memref.alloca() : memref<1x!sycl_local_accessor_1_i64_>
// CHECK-NEXT:             func.call @init(%[[VAL_12]], %[[VAL_0]], %[[VAL_6]], %[[VAL_9]], %[[VAL_11]]) : (memref<1x!sycl_local_accessor_1_i64_>, memref<?xi64, 1>, memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>, memref<?x!sycl_id_1_>) -> ()
// CHECK-NEXT:             gpu.return
// CHECK-NEXT:           }
  gpu.func @k(%ptr: memref<?xi64, 1>,
              %accRange: memref<?x!sycl_range_1_>,
              %memRange: memref<?x!sycl_range_1_>,
              %offset: memref<?x!sycl_id_1_>) kernel {
    %acc = memref.alloca() : memref<1x!sycl_local_accessor_1_i64>
    func.call @init(%acc, %ptr, %accRange, %memRange, %offset)
        : (memref<1x!sycl_local_accessor_1_i64>, memref<?xi64, 1>,
           memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>,
           memref<?x!sycl_id_1_>) -> ()
    gpu.return
  }
}

llvm.func internal @known_range(
  %lambda: !llvm.ptr, %handler: !llvm.ptr) {
  %c1_i32 = llvm.mlir.constant(1 : i32) : i32
  %n = llvm.mlir.constant(512 : i64) : i64
  %range = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::range.1", opaque>
      : (i32) -> !llvm.ptr
  %acc = llvm.alloca %c1_i32 x !llvm.struct<"sycl::_V1::local_accessor", opaque>
      : (i32) -> !llvm.ptr
  sycl.host.constructor(%range, %n)
      {type=!sycl_range_1_}
      : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%acc, %range, %handler)
      {type=!sycl_local_accessor_host}
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.set_captured %lambda[0] = %acc : !llvm.ptr, !llvm.ptr (!sycl_local_accessor_host)
  sycl.host.schedule_kernel %handler -> @kernels1::@k(%acc: !sycl_local_accessor_host)
      : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}
