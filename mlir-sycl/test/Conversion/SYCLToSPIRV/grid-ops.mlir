// RUN: sycl-mlir-opt -convert-sycl-to-spirv %s | FileCheck %s

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64>)>)>

module attributes {gpu.container_module} {
  // CHECK:   gpu.module @kernels {
  gpu.module @kernels {
    // CHECK-DAG:           spirv.GlobalVariable @[[NW:.*]] built_in("NumWorkgroups") : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-DAG:           spirv.GlobalVariable @[[GO:.*]] built_in("GlobalOffset") : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-DAG:           spirv.GlobalVariable @[[WI:.*]] built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-DAG:           spirv.GlobalVariable @[[NWI:.*]] built_in("GlobalSize") : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-DAG:           spirv.GlobalVariable @[[WGS:.*]] built_in("WorkgroupSize") : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-DAG:           spirv.GlobalVariable @[[LII:.*]] built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-DAG:           spirv.GlobalVariable @[[GII:.*]] built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-DAG:           spirv.GlobalVariable @[[SLIID:.*]] built_in("SubgroupLocalInvocationId") : !spirv.ptr<i32, Input>
    // CHECK-DAG:           spirv.GlobalVariable @[[SMS:.*]] built_in("SubgroupMaxSize") : !spirv.ptr<i32, Input>
    // CHECK-DAG:           spirv.GlobalVariable @[[SI:.*]] built_in("SubgroupId") : !spirv.ptr<i32, Input>
    // CHECK-DAG:           spirv.GlobalVariable @[[NS:.*]] built_in("NumSubgroups") : !spirv.ptr<i32, Input>
    // CHECK-DAG:           spirv.GlobalVariable @[[SS:.*]] built_in("SubgroupSize") : !spirv.ptr<i32, Input>

    // CHECK-LABEL:         gpu.func @test_global_offset() kernel
    // CHECK-NEXT:            %[[VAL_0:.*]] = spirv.mlir.addressof @[[GO]] : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-NEXT:            %[[VAL_1:.*]] = spirv.Load "Input" %[[VAL_0]] : vector<3xi32>
    // CHECK-NEXT:            %[[VAL_2:.*]] = memref.alloca() : memref<1x!sycl_id_1_>
    // CHECK-NEXT:            %[[VAL_3:.*]] = arith.constant 0 : index
    // CHECK-NEXT:            %[[VAL_5:.*]] = arith.constant 0 : i32
    // CHECK-NEXT:            %[[VAL_6:.*]] = spirv.CompositeExtract %[[VAL_1]][0 : i32] : vector<3xi32>
    // CHECK-NEXT:            %[[VAL_7:.*]] = arith.extsi %[[VAL_6]] : i32 to i64
    // CHECK-NEXT:            %[[VAL_8:.*]] = sycl.id.get %[[VAL_2]][%[[VAL_5]]] : (memref<1x!sycl_id_1_>, i32) -> memref<1xi64>
    // CHECK-NEXT:            memref.store %[[VAL_7]], %[[VAL_8]]{{\[}}%[[VAL_3]]] : memref<1xi64>
    // CHECK-NEXT:            %[[VAL_4:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_3]]] : memref<1x!sycl_id_1_>
    // CHECK-NEXT:            gpu.return
    // CHECK-NEXT:          }
    gpu.func @test_global_offset() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = sycl.global_offset : !sycl_id_1_
      gpu.return
    }

    // CHECK-LABEL:         gpu.func @test_num_work_groups() kernel
    // CHECK-NEXT:            %[[VAL_24:.*]] = spirv.mlir.addressof @[[NW]] : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-NEXT:            %[[VAL_25:.*]] = spirv.Load "Input" %[[VAL_24]] : vector<3xi32>
    // CHECK-NEXT:            %[[VAL_26:.*]] = memref.alloca() : memref<1x!sycl_range_2_>
    // CHECK-NEXT:            %[[VAL_27:.*]] = arith.constant 0 : index
    // CHECK-NEXT:            %[[VAL_29:.*]] = arith.constant 0 : i32
    // CHECK-NEXT:            %[[VAL_30:.*]] = spirv.CompositeExtract %[[VAL_25]][1 : i32] : vector<3xi32>
    // CHECK-NEXT:            %[[VAL_31:.*]] = arith.extsi %[[VAL_30]] : i32 to i64
    // CHECK-NEXT:            %[[VAL_32:.*]] = sycl.range.get %[[VAL_26]][%[[VAL_29]]] : (memref<1x!sycl_range_2_>, i32) -> memref<2xi64>
    // CHECK-NEXT:            memref.store %[[VAL_31]], %[[VAL_32]]{{\[}}%[[VAL_27]]] : memref<2xi64>
    // CHECK-NEXT:            %[[VAL_33:.*]] = arith.constant 1 : i32
    // CHECK-NEXT:            %[[VAL_34:.*]] = spirv.CompositeExtract %[[VAL_25]][0 : i32] : vector<3xi32>
    // CHECK-NEXT:            %[[VAL_35:.*]] = arith.extsi %[[VAL_34]] : i32 to i64
    // CHECK-NEXT:            %[[VAL_36:.*]] = sycl.range.get %[[VAL_26]][%[[VAL_33]]] : (memref<1x!sycl_range_2_>, i32) -> memref<2xi64>
    // CHECK-NEXT:            memref.store %[[VAL_35]], %[[VAL_36]]{{\[}}%[[VAL_27]]] : memref<2xi64>
    // CHECK-NEXT:            %[[VAL_28:.*]] = memref.load %[[VAL_26]]{{\[}}%[[VAL_27]]] : memref<1x!sycl_range_2_>
    // CHECK-NEXT:            gpu.return
    // CHECK-NEXT:          }
    gpu.func @test_num_work_groups() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = sycl.num_work_groups : !sycl_range_2_
      gpu.return
    }

    // CHECK-LABEL:         gpu.func @test_work_group_id() kernel
    // CHECK-NEXT:            %[[VAL_0:.*]] = spirv.mlir.addressof @[[WI]] : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-NEXT:            %[[VAL_1:.*]] = spirv.Load "Input" %[[VAL_0]] : vector<3xi32>
    // CHECK-NEXT:            %[[VAL_2:.*]] = memref.alloca() : memref<1x!sycl_id_1_>
    // CHECK-NEXT:            %[[VAL_3:.*]] = arith.constant 0 : index
    // CHECK-NEXT:            %[[VAL_5:.*]] = arith.constant 0 : i32
    // CHECK-NEXT:            %[[VAL_6:.*]] = spirv.CompositeExtract %[[VAL_1]][0 : i32] : vector<3xi32>
    // CHECK-NEXT:            %[[VAL_7:.*]] = arith.extsi %[[VAL_6]] : i32 to i64
    // CHECK-NEXT:            %[[VAL_8:.*]] = sycl.id.get %[[VAL_2]][%[[VAL_5]]] : (memref<1x!sycl_id_1_>, i32) -> memref<1xi64>
    // CHECK-NEXT:            memref.store %[[VAL_7]], %[[VAL_8]]{{\[}}%[[VAL_3]]] : memref<1xi64>
    // CHECK-NEXT:            %[[VAL_4:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_3]]] : memref<1x!sycl_id_1_>
    // CHECK-NEXT:            gpu.return
    // CHECK-NEXT:          }
    gpu.func @test_work_group_id() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = sycl.work_group_id : !sycl_id_1_
      gpu.return
    }

    // CHECK-LABEL:         gpu.func @test_num_work_items() kernel
    // CHECK-NEXT:            %[[VAL_0:.*]] = spirv.mlir.addressof @[[NWI]] : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-NEXT:            %[[VAL_1:.*]] = spirv.Load "Input" %[[VAL_0]] : vector<3xi32>
    // CHECK-NEXT:            %[[VAL_2:.*]] = memref.alloca() : memref<1x!sycl_range_1_>
    // CHECK-NEXT:            %[[VAL_3:.*]] = arith.constant 0 : index
    // CHECK-NEXT:            %[[VAL_5:.*]] = arith.constant 0 : i32
    // CHECK-NEXT:            %[[VAL_6:.*]] = spirv.CompositeExtract %[[VAL_1]][0 : i32] : vector<3xi32>
    // CHECK-NEXT:            %[[VAL_7:.*]] = arith.extsi %[[VAL_6]] : i32 to i64
    // CHECK-NEXT:            %[[VAL_8:.*]] = sycl.range.get %[[VAL_2]][%[[VAL_5]]] : (memref<1x!sycl_range_1_>, i32) -> memref<1xi64>
    // CHECK-NEXT:            memref.store %[[VAL_7]], %[[VAL_8]]{{\[}}%[[VAL_3]]] : memref<1xi64>
    // CHECK-NEXT:            %[[VAL_4:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_3]]] : memref<1x!sycl_range_1_>
    // CHECK-NEXT:            gpu.return
    // CHECK-NEXT:          }
    gpu.func @test_num_work_items() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = sycl.num_work_items : !sycl_range_1_
      gpu.return
    }

    // CHECK-LABEL:         gpu.func @test_work_group_size() kernel
    // CHECK-NEXT:            %[[VAL_0:.*]] = spirv.mlir.addressof @[[WGS]] : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-NEXT:            %[[VAL_1:.*]] = spirv.Load "Input" %[[VAL_0]] : vector<3xi32>
    // CHECK-NEXT:            %[[VAL_2:.*]] = memref.alloca() : memref<1x!sycl_range_1_>
    // CHECK-NEXT:            %[[VAL_3:.*]] = arith.constant 0 : index
    // CHECK-NEXT:            %[[VAL_5:.*]] = arith.constant 0 : i32
    // CHECK-NEXT:            %[[VAL_6:.*]] = spirv.CompositeExtract %[[VAL_1]][0 : i32] : vector<3xi32>
    // CHECK-NEXT:            %[[VAL_7:.*]] = arith.extsi %[[VAL_6]] : i32 to i64
    // CHECK-NEXT:            %[[VAL_8:.*]] = sycl.range.get %[[VAL_2]][%[[VAL_5]]] : (memref<1x!sycl_range_1_>, i32) -> memref<1xi64>
    // CHECK-NEXT:            memref.store %[[VAL_7]], %[[VAL_8]]{{\[}}%[[VAL_3]]] : memref<1xi64>
    // CHECK-NEXT:            %[[VAL_4:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_3]]] : memref<1x!sycl_range_1_>
    // CHECK-NEXT:            gpu.return
    // CHECK-NEXT:          }
    gpu.func @test_work_group_size() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = sycl.work_group_size : !sycl_range_1_
      gpu.return
    }

    // CHECK-LABEL:         gpu.func @test_work_group_size_3D() kernel
    // CHECK-NEXT:             %[[VAL_105:.*]] = spirv.mlir.addressof @__spirv_BuiltInWorkgroupSize : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-NEXT:             %[[VAL_106:.*]] = spirv.Load "Input" %[[VAL_105]] : vector<3xi32>
    // CHECK-NEXT:             %[[VAL_107:.*]] = memref.alloca() : memref<1x!sycl_range_3_>
    // CHECK-NEXT:             %[[VAL_108:.*]] = arith.constant 0 : index
    // CHECK-NEXT:             %[[VAL_109:.*]] = arith.constant 0 : i32
    // CHECK-NEXT:             %[[VAL_110:.*]] = spirv.CompositeExtract %[[VAL_106]][2 : i32] : vector<3xi32>
    // CHECK-NEXT:             %[[VAL_111:.*]] = arith.extsi %[[VAL_110]] : i32 to i64
    // CHECK-NEXT:             %[[VAL_112:.*]] = sycl.range.get %[[VAL_107]]{{\[}}%[[VAL_109]]] : (memref<1x!sycl_range_3_>, i32) -> memref<3xi64>
    // CHECK-NEXT:             memref.store %[[VAL_111]], %[[VAL_112]]{{\[}}%[[VAL_108]]] : memref<3xi64>
    // CHECK-NEXT:             %[[VAL_113:.*]] = arith.constant 1 : i32
    // CHECK-NEXT:             %[[VAL_114:.*]] = spirv.CompositeExtract %[[VAL_106]][1 : i32] : vector<3xi32>
    // CHECK-NEXT:             %[[VAL_115:.*]] = arith.extsi %[[VAL_114]] : i32 to i64
    // CHECK-NEXT:             %[[VAL_116:.*]] = sycl.range.get %[[VAL_107]]{{\[}}%[[VAL_113]]] : (memref<1x!sycl_range_3_>, i32) -> memref<3xi64>
    // CHECK-NEXT:             memref.store %[[VAL_115]], %[[VAL_116]]{{\[}}%[[VAL_108]]] : memref<3xi64>
    // CHECK-NEXT:             %[[VAL_117:.*]] = arith.constant 2 : i32
    // CHECK-NEXT:             %[[VAL_118:.*]] = spirv.CompositeExtract %[[VAL_106]][0 : i32] : vector<3xi32>
    // CHECK-NEXT:             %[[VAL_119:.*]] = arith.extsi %[[VAL_118]] : i32 to i64
    // CHECK-NEXT:             %[[VAL_120:.*]] = sycl.range.get %[[VAL_107]]{{\[}}%[[VAL_117]]] : (memref<1x!sycl_range_3_>, i32) -> memref<3xi64>
    // CHECK-NEXT:             memref.store %[[VAL_119]], %[[VAL_120]]{{\[}}%[[VAL_108]]] : memref<3xi64>
    // CHECK-NEXT:             %[[VAL_121:.*]] = memref.load %[[VAL_107]]{{\[}}%[[VAL_108]]] : memref<1x!sycl_range_3_>
    // CHECK-NEXT:             gpu.return
    // CHECK-NEXT:           }
    gpu.func @test_work_group_size_3D() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = sycl.work_group_size : !sycl_range_3_
      gpu.return
    }

    // CHECK-LABEL:         gpu.func @test_local_id() kernel
    // CHECK-NEXT:            %[[VAL_0:.*]] = spirv.mlir.addressof @[[LII]] : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-NEXT:            %[[VAL_1:.*]] = spirv.Load "Input" %[[VAL_0]] : vector<3xi32>
    // CHECK-NEXT:            %[[VAL_2:.*]] = memref.alloca() : memref<1x!sycl_id_1_>
    // CHECK-NEXT:            %[[VAL_3:.*]] = arith.constant 0 : index
    // CHECK-NEXT:            %[[VAL_5:.*]] = arith.constant 0 : i32
    // CHECK-NEXT:            %[[VAL_6:.*]] = spirv.CompositeExtract %[[VAL_1]][0 : i32] : vector<3xi32>
    // CHECK-NEXT:            %[[VAL_7:.*]] = arith.extsi %[[VAL_6]] : i32 to i64
    // CHECK-NEXT:            %[[VAL_8:.*]] = sycl.id.get %[[VAL_2]][%[[VAL_5]]] : (memref<1x!sycl_id_1_>, i32) -> memref<1xi64>
    // CHECK-NEXT:            memref.store %[[VAL_7]], %[[VAL_8]]{{\[}}%[[VAL_3]]] : memref<1xi64>
    // CHECK-NEXT:            %[[VAL_4:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_3]]] : memref<1x!sycl_id_1_>
    // CHECK-NEXT:            gpu.return
    // CHECK-NEXT:          }
    gpu.func @test_local_id() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = sycl.local_id : !sycl_id_1_
      gpu.return
    }

    // CHECK-LABEL:         gpu.func @test_global_id() kernel
    // CHECK-NEXT:            %[[VAL_0:.*]] = spirv.mlir.addressof @[[GII]] : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-NEXT:            %[[VAL_1:.*]] = spirv.Load "Input" %[[VAL_0]] : vector<3xi32>
    // CHECK-NEXT:            %[[VAL_2:.*]] = memref.alloca() : memref<1x!sycl_id_1_>
    // CHECK-NEXT:            %[[VAL_3:.*]] = arith.constant 0 : index
    // CHECK-NEXT:            %[[VAL_5:.*]] = arith.constant 0 : i32
    // CHECK-NEXT:            %[[VAL_6:.*]] = spirv.CompositeExtract %[[VAL_1]][0 : i32] : vector<3xi32>
    // CHECK-NEXT:            %[[VAL_7:.*]] = arith.extsi %[[VAL_6]] : i32 to i64
    // CHECK-NEXT:            %[[VAL_8:.*]] = sycl.id.get %[[VAL_2]][%[[VAL_5]]] : (memref<1x!sycl_id_1_>, i32) -> memref<1xi64>
    // CHECK-NEXT:            memref.store %[[VAL_7]], %[[VAL_8]]{{\[}}%[[VAL_3]]] : memref<1xi64>
    // CHECK-NEXT:            %[[VAL_4:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_3]]] : memref<1x!sycl_id_1_>
    // CHECK-NEXT:            gpu.return
    // CHECK-NEXT:          }
    gpu.func @test_global_id() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = sycl.global_id : !sycl_id_1_
      gpu.return
    }

    // CHECK-LABEL:         gpu.func @test_sub_group_max_size() kernel
    // CHECK-NEXT:            %[[VAL_52:.*]] = spirv.mlir.addressof @[[SMS]] : !spirv.ptr<i32, Input>
    // CHECK-NEXT:            %[[VAL_53:.*]] = spirv.Load "Input" %[[VAL_52]] : i32
    // CHECK-NEXT:            gpu.return
    // CHECK-NEXT:          }
    gpu.func @test_sub_group_max_size() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = sycl.sub_group_max_size : i32
      gpu.return
    }

    // CHECK-LABEL:         gpu.func @test_sub_group_local_id() kernel
    // CHECK-NEXT:            %[[VAL_54:.*]] = spirv.mlir.addressof @[[SLIID]] : !spirv.ptr<i32, Input>
    // CHECK-NEXT:            %[[VAL_55:.*]] = spirv.Load "Input" %[[VAL_54]] : i32
    // CHECK-NEXT:            gpu.return
    // CHECK-NEXT:          }
    gpu.func @test_sub_group_local_id() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = sycl.sub_group_local_id : i32
      gpu.return
    }

    // CHECK-LABEL:         gpu.func @test_sub_group_id() kernel
    // CHECK-NEXT:            %[[VAL_54:.*]] = spirv.mlir.addressof @[[SI]] : !spirv.ptr<i32, Input>
    // CHECK-NEXT:            %[[VAL_55:.*]] = spirv.Load "Input" %[[VAL_54]] : i32
    // CHECK-NEXT:            gpu.return
    // CHECK-NEXT:          }
    gpu.func @test_sub_group_id() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = sycl.sub_group_id : i32
      gpu.return
    }

    // CHECK-LABEL:         gpu.func @test_num_sub_groups() kernel
    // CHECK-NEXT:            %[[VAL_54:.*]] = spirv.mlir.addressof @[[NS]] : !spirv.ptr<i32, Input>
    // CHECK-NEXT:            %[[VAL_55:.*]] = spirv.Load "Input" %[[VAL_54]] : i32
    // CHECK-NEXT:            gpu.return
    // CHECK-NEXT:          }
    gpu.func @test_num_sub_groups() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = sycl.num_sub_groups : i32
      gpu.return
    }

    // CHECK-LABEL:         gpu.func @test_sub_group_size() kernel
    // CHECK-NEXT:            %[[VAL_54:.*]] = spirv.mlir.addressof @[[SS]] : !spirv.ptr<i32, Input>
    // CHECK-NEXT:            %[[VAL_55:.*]] = spirv.Load "Input" %[[VAL_54]] : i32
    // CHECK-NEXT:            gpu.return
    // CHECK-NEXT:          }
    gpu.func @test_sub_group_size() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = sycl.sub_group_size : i32
      gpu.return
    }
  }
}
