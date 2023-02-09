// RUN: sycl-mlir-opt -convert-sycl-to-spirv %s -o - | FileCheck %s

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>

module attributes {gpu.container_module} {
  // CHECK:   gpu.module @kernels {
  gpu.module @kernels {
    // CHECK-DAG:           spirv.GlobalVariable @[[SLIID:.*]] built_in("SubgroupLocalInvocationId") : !spirv.ptr<i32, Input>
    // CHECK-DAG:           spirv.GlobalVariable @[[SMS:.*]] built_in("SubgroupMaxSize") : !spirv.ptr<i32, Input>
    // CHECK-DAG:           spirv.GlobalVariable @[[NW:.*]] built_in("NumWorkgroups") : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-DAG:           spirv.GlobalVariable @[[GO:.*]] built_in("GlobalOffset") : !spirv.ptr<vector<3xi32>, Input>

    // CHECK-LABEL:         gpu.func @test_global_offset() kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK-NEXT:            %[[VAL_0:.*]] = spirv.mlir.addressof @[[GO]] : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-NEXT             %[[VAL_1:.*]] = spirv.Load "Input" %[[VAL_0]] : vector<3xi32>
    // CHECK-NEXT             %[[VAL_2:.*]] = memref.alloca() : memref<1x!sycl_id_1_>
    // CHECK-NEXT             %[[VAL_3:.*]] = arith.constant 0 : index
    // CHECK-NEXT             %[[VAL_4:.*]] = affine.load %[[VAL_2]]{{\[}}%[[VAL_3]]] : memref<1x!sycl_id_1_>
    // CHECK-NEXT             %[[VAL_5:.*]] = arith.constant 0 : i32
    // CHECK-NEXT             %[[VAL_6:.*]] = spirv.CompositeExtract %[[VAL_1]][0 : i32] : vector<3xi32>
    // CHECK-NEXT             %[[VAL_7:.*]] = arith.extsi %[[VAL_6]] : i32 to i64
    // CHECK-NEXT             %[[VAL_8:.*]] = "sycl.id.get"(%[[VAL_4]], %[[VAL_5]]) {ArgumentTypes = [memref<1x!sycl_id_1_, 4>, i32], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @id} : (!sycl_id_1_, i32) -> memref<1xi64, 4>
    // CHECK-NEXT             affine.store %[[VAL_7]], %[[VAL_8]]{{\[}}%[[VAL_3]]] : memref<1xi64, 4>
    // CHECK-NEXT             gpu.return
    // CHECK-NEXT           }
    gpu.func @test_global_offset() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      %0 = sycl.global_offset() : () -> !sycl_id_1_
      gpu.return
    }

    // CHECK-LABEL:         gpu.func @test_global_offset_dim(
    // CHECK-SAME:                                           %[[VAL_9:.*]]: i32) kernel
    // CHECK-NEXT             %[[VAL_10:.*]] = spirv.mlir.addressof @[[GO]] : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-NEXT             %[[VAL_11:.*]] = spirv.Load "Input" %[[VAL_10]] : vector<3xi32>
    // CHECK-NEXT             %[[VAL_12:.*]] = memref.alloca() : memref<3xindex>
    // CHECK-NEXT             %[[VAL_13:.*]] = arith.constant 0 : index
    // CHECK-NEXT             %[[VAL_14:.*]] = spirv.CompositeExtract %[[VAL_11]][0 : i32] : vector<3xi32>
    // CHECK-NEXT             %[[VAL_15:.*]] = arith.index_cast %[[VAL_14]] : i32 to index
    // CHECK-NEXT             affine.store %[[VAL_15]], %[[VAL_12]]{{\[}}%[[VAL_13]]] : memref<3xindex>
    // CHECK-NEXT             %[[VAL_16:.*]] = arith.constant 1 : index
    // CHECK-NEXT             %[[VAL_17:.*]] = spirv.CompositeExtract %[[VAL_11]][1 : i32] : vector<3xi32>
    // CHECK-NEXT             %[[VAL_18:.*]] = arith.index_cast %[[VAL_17]] : i32 to index
    // CHECK-NEXT             affine.store %[[VAL_18]], %[[VAL_12]]{{\[}}%[[VAL_16]]] : memref<3xindex>
    // CHECK-NEXT             %[[VAL_19:.*]] = arith.constant 2 : index
    // CHECK-NEXT             %[[VAL_20:.*]] = spirv.CompositeExtract %[[VAL_11]][2 : i32] : vector<3xi32>
    // CHECK-NEXT             %[[VAL_21:.*]] = arith.index_cast %[[VAL_20]] : i32 to index
    // CHECK-NEXT             affine.store %[[VAL_21]], %[[VAL_12]]{{\[}}%[[VAL_19]]] : memref<3xindex>
    // CHECK-NEXT             %[[VAL_22:.*]] = arith.index_cast %[[VAL_9]] : i32 to index
    // CHECK-NEXT             %[[VAL_23:.*]] = affine.load %[[VAL_12]]{{\[}}%[[VAL_22]]] : memref<3xindex>
    // CHECK-NEXT             gpu.return
    // CHECK-NEXT           }
    gpu.func @test_global_offset_dim(%i: i32) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      %0 = sycl.global_offset(%i) : (i32) -> index
      gpu.return
    }

    // CHECK-LABEL:         gpu.func @test_num_work_groups() kernel
    // CHECK-NEXT             %[[VAL_24:.*]] = spirv.mlir.addressof @[[NW]] : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-NEXT             %[[VAL_25:.*]] = spirv.Load "Input" %[[VAL_24]] : vector<3xi32>
    // CHECK-NEXT             %[[VAL_26:.*]] = memref.alloca() : memref<1x!sycl_range_2_>
    // CHECK-NEXT             %[[VAL_27:.*]] = arith.constant 0 : index
    // CHECK-NEXT             %[[VAL_28:.*]] = affine.load %[[VAL_26]]{{\[}}%[[VAL_27]]] : memref<1x!sycl_range_2_>
    // CHECK-NEXT             %[[VAL_29:.*]] = arith.constant 0 : i32
    // CHECK-NEXT             %[[VAL_30:.*]] = spirv.CompositeExtract %[[VAL_25]][0 : i32] : vector<3xi32>
    // CHECK-NEXT             %[[VAL_31:.*]] = arith.extsi %[[VAL_30]] : i32 to i64
    // CHECK-NEXT             %[[VAL_32:.*]] = "sycl.range.get"(%[[VAL_28]], %[[VAL_29]]) {ArgumentTypes = [memref<1x!sycl_range_2_, 4>, i32], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @range} : (!sycl_range_2_, i32) -> memref<2xi64, 4>
    // CHECK-NEXT             affine.store %[[VAL_31]], %[[VAL_32]]{{\[}}%[[VAL_27]]] : memref<2xi64, 4>
    // CHECK-NEXT             %[[VAL_33:.*]] = arith.constant 1 : i32
    // CHECK-NEXT             %[[VAL_34:.*]] = spirv.CompositeExtract %[[VAL_25]][1 : i32] : vector<3xi32>
    // CHECK-NEXT             %[[VAL_35:.*]] = arith.extsi %[[VAL_34]] : i32 to i64
    // CHECK-NEXT             %[[VAL_36:.*]] = "sycl.range.get"(%[[VAL_28]], %[[VAL_33]]) {ArgumentTypes = [memref<1x!sycl_range_2_, 4>, i32], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @range} : (!sycl_range_2_, i32) -> memref<2xi64, 4>
    // CHECK-NEXT             affine.store %[[VAL_35]], %[[VAL_36]]{{\[}}%[[VAL_27]]] : memref<2xi64, 4>
    // CHECK-NEXT             gpu.return
    // CHECK-NEXT           }
    gpu.func @test_num_work_groups() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      %0 = sycl.num_work_groups() : () -> !sycl_range_2_
      gpu.return
    }

    // CHECK-LABEL:         gpu.func @test_num_work_groups_dim(
    // CHECK-SAME:                                             %[[VAL_37:.*]]: i32) kernel
    // CHECK-NEXT             %[[VAL_38:.*]] = spirv.mlir.addressof @[[NW]] : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-NEXT             %[[VAL_39:.*]] = spirv.Load "Input" %[[VAL_38]] : vector<3xi32>
    // CHECK-NEXT             %[[VAL_40:.*]] = memref.alloca() : memref<3xindex>
    // CHECK-NEXT             %[[VAL_41:.*]] = arith.constant 0 : index
    // CHECK-NEXT             %[[VAL_42:.*]] = spirv.CompositeExtract %[[VAL_39]][0 : i32] : vector<3xi32>
    // CHECK-NEXT             %[[VAL_43:.*]] = arith.index_cast %[[VAL_42]] : i32 to index
    // CHECK-NEXT             affine.store %[[VAL_43]], %[[VAL_40]]{{\[}}%[[VAL_41]]] : memref<3xindex>
    // CHECK-NEXT             %[[VAL_44:.*]] = arith.constant 1 : index
    // CHECK-NEXT             %[[VAL_45:.*]] = spirv.CompositeExtract %[[VAL_39]][1 : i32] : vector<3xi32>
    // CHECK-NEXT             %[[VAL_46:.*]] = arith.index_cast %[[VAL_45]] : i32 to index
    // CHECK-NEXT             affine.store %[[VAL_46]], %[[VAL_40]]{{\[}}%[[VAL_44]]] : memref<3xindex>
    // CHECK-NEXT             %[[VAL_47:.*]] = arith.constant 2 : index
    // CHECK-NEXT             %[[VAL_48:.*]] = spirv.CompositeExtract %[[VAL_39]][2 : i32] : vector<3xi32>
    // CHECK-NEXT             %[[VAL_49:.*]] = arith.index_cast %[[VAL_48]] : i32 to index
    // CHECK-NEXT             affine.store %[[VAL_49]], %[[VAL_40]]{{\[}}%[[VAL_47]]] : memref<3xindex>
    // CHECK-NEXT             %[[VAL_50:.*]] = arith.index_cast %[[VAL_37]] : i32 to index
    // CHECK-NEXT             %[[VAL_51:.*]] = affine.load %[[VAL_40]]{{\[}}%[[VAL_50]]] : memref<3xindex>
    // CHECK-NEXT             gpu.return
    // CHECK-NEXT           }
    gpu.func @test_num_work_groups_dim(%i: i32) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      %0 = sycl.num_work_groups(%i) : (i32) -> index
      gpu.return
    }

    // CHECK-LABEL:         gpu.func @test_sub_group_max_size() kernel
    // CHECK-NEXT             %[[VAL_52:.*]] = spirv.mlir.addressof @[[SMS]] : !spirv.ptr<i32, Input>
    // CHECK-NEXT             %[[VAL_53:.*]] = spirv.Load "Input" %[[VAL_52]] : i32
    // CHECK-NEXT             gpu.return
    // CHECK-NEXT           }
    gpu.func @test_sub_group_max_size() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      %0 = sycl.sub_group_max_size : () -> i32
      gpu.return
    }

    // CHECK-LABEL:         gpu.func @test_sub_group_local_id() kernel
    // CHECK-NEXT             %[[VAL_54:.*]] = spirv.mlir.addressof @[[SLIID]] : !spirv.ptr<i32, Input>
    // CHECK-NEXT             %[[VAL_55:.*]] = spirv.Load "Input" %[[VAL_54]] : i32
    // CHECK-NEXT             gpu.return
    // CHECK-NEXT           }
    gpu.func @test_sub_group_local_id() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      %0 = sycl.sub_group_local_id : () -> i32
      gpu.return
    }
  }
}
