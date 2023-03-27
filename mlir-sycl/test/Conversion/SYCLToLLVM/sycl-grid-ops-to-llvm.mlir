// RUN: sycl-mlir-opt -convert-sycl-to-llvm %s -o - | FileCheck %s

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

module attributes {gpu.container_module} {
  // CHECK:   gpu.module @kernels {
    //CHECK-DAG:      llvm.mlir.global external constant @__builtin_var_SubgroupLocalInvocationId__() {addr_space = 0 : i32} : i64
    //CHECK-DAG:      llvm.mlir.global external constant @__builtin_var_SubgroupMaxSize__() {addr_space = 0 : i32} : i64
    //CHECK-DAG:      llvm.mlir.global external constant @__builtin_var_GlobalOffset__() {addr_space = 0 : i32} : vector<3xi64>
    //CHECK-DAG:      llvm.mlir.global external constant @__builtin_var_SubgroupId__() {addr_space = 0 : i32} : i64
    //CHECK-DAG:      llvm.mlir.global external constant @__builtin_var_SubgroupSize__() {addr_space = 0 : i32} : i64
    //CHECK-DAG:      llvm.mlir.global external constant @__builtin_var_NumSubgroups__() {addr_space = 0 : i32} : i64
    //CHECK-DAG:      llvm.mlir.global external constant @__builtin_var_WorkgroupId__() {addr_space = 0 : i32} : vector<3xi64>
    //CHECK-DAG:      llvm.mlir.global external constant @__builtin_var_WorkgroupSize__() {addr_space = 0 : i32} : vector<3xi64>
    //CHECK-DAG:      llvm.mlir.global external constant @__builtin_var_LocalInvocationId__() {addr_space = 0 : i32} : vector<3xi64>
    //CHECK-DAG:      llvm.mlir.global external constant @__builtin_var_GlobalInvocationId__() {addr_space = 0 : i32} : vector<3xi64>
    //CHECK-DAG:      llvm.mlir.global external constant @__builtin_var_NumWorkgroups__() {addr_space = 0 : i32} : vector<3xi64>
    //CHECK-DAG:      llvm.mlir.global external constant @__builtin_var_GlobalSize__() {addr_space = 0 : i32} : vector<3xi64>
  gpu.module @kernels {
    // CHECK-LABEL:   llvm.func @test_num_work_items() -> !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> {
    // CHECK:             %[[VAL_0:.*]] = llvm.mlir.addressof @__builtin_var_GlobalSize__ : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_1:.*]] = llvm.load %[[VAL_0]] : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_2:.*]] = llvm.mlir.null : !llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK:             %[[VAL_3:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK:             %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_2]]{{\[}}%[[VAL_3]]] : (!llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>, i64) -> !llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK:             %[[VAL_5:.*]] = llvm.ptrtoint %[[VAL_4]] : !llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>> to i64
    // CHECK:             %[[VAL_6:.*]] = llvm.alloca %[[VAL_5]] x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> : (i64) -> !llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_7:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_8:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:         %[[VAL_9:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:             %[[VAL_10:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_9]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_6]][0, 0, 0, %[[VAL_8]]] : (!llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>, i32) -> !llvm.ptr<i64>
    // CHECK:             %[[VAL_12:.*]] = llvm.getelementptr %[[VAL_11]]{{\[}}%[[VAL_7]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    // CHECK:             llvm.store %[[VAL_10]], %[[VAL_12]] : !llvm.ptr<i64>
    // CHECK:             %[[VAL_13:.*]] = llvm.getelementptr %[[VAL_6]]{{\[}}%[[VAL_7]]] : (!llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>, i64) -> !llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK:             %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK:             llvm.return %[[VAL_14]] : !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>
    // CHECK-NEXT:      }
    func.func @test_num_work_items() -> !sycl_range_1_ {
      %0 = sycl.num_work_items : !sycl_range_1_
      return %0 : !sycl_range_1_
    }

    // CHECK-LABEL:     llvm.func @test_num_work_items_dim(
    // CHECK-SAME:                                         %[[VAL_15:.*]]: i32) -> i64 {
    // CHECK:             %[[VAL_16:.*]] = llvm.mlir.addressof @__builtin_var_GlobalSize__ : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_17:.*]] = llvm.load %[[VAL_16]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_18:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
    // CHECK-DAG:         %[[VAL_19:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:             %[[VAL_20:.*]] = llvm.extractelement %[[VAL_17]]{{\[}}%[[VAL_19]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_21:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK:             %[[VAL_22:.*]] = llvm.insertelement %[[VAL_20]], %[[VAL_18]]{{\[}}%[[VAL_21]] : i64] : vector<3xi64>
    // CHECK:             %[[VAL_23:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:             %[[VAL_24:.*]] = llvm.extractelement %[[VAL_17]]{{\[}}%[[VAL_23]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_25:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK:             %[[VAL_26:.*]] = llvm.insertelement %[[VAL_24]], %[[VAL_22]]{{\[}}%[[VAL_25]] : i64] : vector<3xi64>
    // CHECK:             %[[VAL_27:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:             %[[VAL_28:.*]] = llvm.extractelement %[[VAL_17]]{{\[}}%[[VAL_27]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_29:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:             %[[VAL_30:.*]] = llvm.insertelement %[[VAL_28]], %[[VAL_26]]{{\[}}%[[VAL_29]] : i64] : vector<3xi64>
    // CHECK:             %[[VAL_31:.*]] = llvm.extractelement %[[VAL_30]]{{\[}}%[[VAL_15]] : i32] : vector<3xi64>
    // CHECK:             llvm.return %[[VAL_31]] : i64
    // CHECK:           }
    func.func @test_num_work_items_dim(%i: i32) -> index {
      %0 = sycl.num_work_items %i : index
      return %0 : index
    }

    // CHECK-LABEL:     llvm.func @test_global_id() -> !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)> {
    // CHECK:             %[[VAL_32:.*]] = llvm.mlir.addressof @__builtin_var_GlobalInvocationId__ : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_33:.*]] = llvm.load %[[VAL_32]] : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_34:.*]] = llvm.mlir.null : !llvm.ptr<struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>
    // CHECK:             %[[VAL_35:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK:             %[[VAL_36:.*]] = llvm.getelementptr %[[VAL_34]]{{\[}}%[[VAL_35]]] : (!llvm.ptr<struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>, i64) -> !llvm.ptr<struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>
    // CHECK:             %[[VAL_37:.*]] = llvm.ptrtoint %[[VAL_36]] : !llvm.ptr<struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>> to i64
    // CHECK:             %[[VAL_38:.*]] = llvm.alloca %[[VAL_37]] x !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)> : (i64) -> !llvm.ptr<struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_39:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_40:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:         %[[VAL_41:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:             %[[VAL_42:.*]] = llvm.extractelement %[[VAL_33]]{{\[}}%[[VAL_41]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_43:.*]] = llvm.getelementptr inbounds %[[VAL_38]][0, 0, 0, %[[VAL_40]]] : (!llvm.ptr<struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>, i32) -> !llvm.ptr<i64>
    // CHECK:             %[[VAL_44:.*]] = llvm.getelementptr %[[VAL_43]]{{\[}}%[[VAL_39]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    // CHECK:             llvm.store %[[VAL_42]], %[[VAL_44]] : !llvm.ptr<i64>
    // CHECK-DAG:         %[[VAL_45:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:         %[[VAL_46:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:             %[[VAL_47:.*]] = llvm.extractelement %[[VAL_33]]{{\[}}%[[VAL_46]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_48:.*]] = llvm.getelementptr inbounds %[[VAL_38]][0, 0, 0, %[[VAL_45]]] : (!llvm.ptr<struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>, i32) -> !llvm.ptr<i64>
    // CHECK:             %[[VAL_49:.*]] = llvm.getelementptr %[[VAL_48]]{{\[}}%[[VAL_39]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    // CHECK:             llvm.store %[[VAL_47]], %[[VAL_49]] : !llvm.ptr<i64>
    // CHECK:             %[[VAL_50:.*]] = llvm.getelementptr %[[VAL_38]]{{\[}}%[[VAL_39]]] : (!llvm.ptr<struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>, i64) -> !llvm.ptr<struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>
    // CHECK:             %[[VAL_51:.*]] = llvm.load %[[VAL_50]] : !llvm.ptr<struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>
    // CHECK:             llvm.return %[[VAL_51]] : !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>
    // CHECK:           }
    func.func @test_global_id() -> !sycl_id_2_ {
      %0 = sycl.global_id : !sycl_id_2_
      return %0 : !sycl_id_2_
    }

    // CHECK-LABEL:     llvm.func @test_global_id_dim(
    // CHECK-SAME:                                    %[[VAL_52:.*]]: i32) -> i64 {
    // CHECK:             %[[VAL_53:.*]] = llvm.mlir.addressof @__builtin_var_GlobalInvocationId__ : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_54:.*]] = llvm.load %[[VAL_53]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_55:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
    // CHECK-DAG:         %[[VAL_56:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:             %[[VAL_57:.*]] = llvm.extractelement %[[VAL_54]]{{\[}}%[[VAL_56]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_58:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK:             %[[VAL_59:.*]] = llvm.insertelement %[[VAL_57]], %[[VAL_55]]{{\[}}%[[VAL_58]] : i64] : vector<3xi64>
    // CHECK:             %[[VAL_60:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:             %[[VAL_61:.*]] = llvm.extractelement %[[VAL_54]]{{\[}}%[[VAL_60]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_62:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK:             %[[VAL_63:.*]] = llvm.insertelement %[[VAL_61]], %[[VAL_59]]{{\[}}%[[VAL_62]] : i64] : vector<3xi64>
    // CHECK:             %[[VAL_64:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:             %[[VAL_65:.*]] = llvm.extractelement %[[VAL_54]]{{\[}}%[[VAL_64]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_66:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:             %[[VAL_67:.*]] = llvm.insertelement %[[VAL_65]], %[[VAL_63]]{{\[}}%[[VAL_66]] : i64] : vector<3xi64>
    // CHECK:             %[[VAL_68:.*]] = llvm.extractelement %[[VAL_67]]{{\[}}%[[VAL_52]] : i32] : vector<3xi64>
    // CHECK:             llvm.return %[[VAL_68]] : i64
    // CHECK:           }
    func.func @test_global_id_dim(%i: i32) -> index {
      %0 = sycl.global_id %i : index
      return %0 : index
    }

    // CHECK-LABEL:     llvm.func @test_local_id() -> !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {
    // CHECK:             %[[VAL_69:.*]] = llvm.mlir.addressof @__builtin_var_LocalInvocationId__ : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_70:.*]] = llvm.load %[[VAL_69]] : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_71:.*]] = llvm.mlir.null : !llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
    // CHECK:             %[[VAL_72:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK:             %[[VAL_73:.*]] = llvm.getelementptr %[[VAL_71]]{{\[}}%[[VAL_72]]] : (!llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>, i64) -> !llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
    // CHECK:             %[[VAL_74:.*]] = llvm.ptrtoint %[[VAL_73]] : !llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>> to i64
    // CHECK:             %[[VAL_75:.*]] = llvm.alloca %[[VAL_74]] x !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> : (i64) -> !llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_76:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_77:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:         %[[VAL_78:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:             %[[VAL_79:.*]] = llvm.extractelement %[[VAL_70]]{{\[}}%[[VAL_78]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_80:.*]] = llvm.getelementptr inbounds %[[VAL_75]][0, 0, 0, %[[VAL_77]]] : (!llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>, i32) -> !llvm.ptr<i64>
    // CHECK:             %[[VAL_81:.*]] = llvm.getelementptr %[[VAL_80]]{{\[}}%[[VAL_76]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    // CHECK:             llvm.store %[[VAL_79]], %[[VAL_81]] : !llvm.ptr<i64>
    // CHECK-DAG:         %[[VAL_82:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:         %[[VAL_83:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:             %[[VAL_84:.*]] = llvm.extractelement %[[VAL_70]]{{\[}}%[[VAL_83]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_85:.*]] = llvm.getelementptr inbounds %[[VAL_75]][0, 0, 0, %[[VAL_82]]] : (!llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>, i32) -> !llvm.ptr<i64>
    // CHECK:             %[[VAL_86:.*]] = llvm.getelementptr %[[VAL_85]]{{\[}}%[[VAL_76]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    // CHECK:             llvm.store %[[VAL_84]], %[[VAL_86]] : !llvm.ptr<i64>
    // CHECK-DAG:         %[[VAL_87:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:         %[[VAL_88:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:             %[[VAL_89:.*]] = llvm.extractelement %[[VAL_70]]{{\[}}%[[VAL_88]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_90:.*]] = llvm.getelementptr inbounds %[[VAL_75]][0, 0, 0, %[[VAL_87]]] : (!llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>, i32) -> !llvm.ptr<i64>
    // CHECK:             %[[VAL_91:.*]] = llvm.getelementptr %[[VAL_90]]{{\[}}%[[VAL_76]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    // CHECK:             llvm.store %[[VAL_89]], %[[VAL_91]] : !llvm.ptr<i64>
    // CHECK:             %[[VAL_92:.*]] = llvm.getelementptr %[[VAL_75]]{{\[}}%[[VAL_76]]] : (!llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>, i64) -> !llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
    // CHECK:             %[[VAL_93:.*]] = llvm.load %[[VAL_92]] : !llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
    // CHECK:             llvm.return %[[VAL_93]] : !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
    // CHECK:           }
    func.func @test_local_id() -> !sycl_id_3_ {
      %0 = sycl.local_id : !sycl_id_3_
      return %0 : !sycl_id_3_
    }

    // CHECK-LABEL:     llvm.func @test_local_id_dim(
    // CHECK-SAME:                                   %[[VAL_94:.*]]: i32) -> i64 {
    // CHECK:             %[[VAL_95:.*]] = llvm.mlir.addressof @__builtin_var_LocalInvocationId__ : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_96:.*]] = llvm.load %[[VAL_95]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_97:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
    // CHECK-DAG:         %[[VAL_98:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:             %[[VAL_99:.*]] = llvm.extractelement %[[VAL_96]]{{\[}}%[[VAL_98]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_100:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK:             %[[VAL_101:.*]] = llvm.insertelement %[[VAL_99]], %[[VAL_97]]{{\[}}%[[VAL_100]] : i64] : vector<3xi64>
    // CHECK:             %[[VAL_102:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:             %[[VAL_103:.*]] = llvm.extractelement %[[VAL_96]]{{\[}}%[[VAL_102]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_104:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK:             %[[VAL_105:.*]] = llvm.insertelement %[[VAL_103]], %[[VAL_101]]{{\[}}%[[VAL_104]] : i64] : vector<3xi64>
    // CHECK:             %[[VAL_106:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:             %[[VAL_107:.*]] = llvm.extractelement %[[VAL_96]]{{\[}}%[[VAL_106]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_108:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:             %[[VAL_109:.*]] = llvm.insertelement %[[VAL_107]], %[[VAL_105]]{{\[}}%[[VAL_108]] : i64] : vector<3xi64>
    // CHECK:             %[[VAL_110:.*]] = llvm.extractelement %[[VAL_109]]{{\[}}%[[VAL_94]] : i32] : vector<3xi64>
    // CHECK:             llvm.return %[[VAL_110]] : i64
    // CHECK:           }
    func.func @test_local_id_dim(%i: i32) -> index {
      %0 = sycl.local_id %i : index
      return %0 : index
    }

    // CHECK-LABEL:     llvm.func @test_work_group_size() -> !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {
    // CHECK:             %[[VAL_111:.*]] = llvm.mlir.addressof @__builtin_var_WorkgroupSize__ : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_112:.*]] = llvm.load %[[VAL_111]] : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_113:.*]] = llvm.mlir.null : !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
    // CHECK:             %[[VAL_114:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK:             %[[VAL_115:.*]] = llvm.getelementptr %[[VAL_113]]{{\[}}%[[VAL_114]]] : (!llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>, i64) -> !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
    // CHECK:             %[[VAL_116:.*]] = llvm.ptrtoint %[[VAL_115]] : !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>> to i64
    // CHECK:             %[[VAL_117:.*]] = llvm.alloca %[[VAL_116]] x !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> : (i64) -> !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_118:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_119:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:         %[[VAL_120:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:             %[[VAL_121:.*]] = llvm.extractelement %[[VAL_112]]{{\[}}%[[VAL_120]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_122:.*]] = llvm.getelementptr inbounds %[[VAL_117]][0, 0, 0, %[[VAL_119]]] : (!llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>, i32) -> !llvm.ptr<i64>
    // CHECK:             %[[VAL_123:.*]] = llvm.getelementptr %[[VAL_122]]{{\[}}%[[VAL_118]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    // CHECK:             llvm.store %[[VAL_121]], %[[VAL_123]] : !llvm.ptr<i64>
    // CHECK-DAG:         %[[VAL_124:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:         %[[VAL_125:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:             %[[VAL_126:.*]] = llvm.extractelement %[[VAL_112]]{{\[}}%[[VAL_125]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_127:.*]] = llvm.getelementptr inbounds %[[VAL_117]][0, 0, 0, %[[VAL_124]]] : (!llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>, i32) -> !llvm.ptr<i64>
    // CHECK:             %[[VAL_128:.*]] = llvm.getelementptr %[[VAL_127]]{{\[}}%[[VAL_118]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    // CHECK:             llvm.store %[[VAL_126]], %[[VAL_128]] : !llvm.ptr<i64>
    // CHECK-DAG:         %[[VAL_129:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:         %[[VAL_130:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:             %[[VAL_131:.*]] = llvm.extractelement %[[VAL_112]]{{\[}}%[[VAL_130]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_132:.*]] = llvm.getelementptr inbounds %[[VAL_117]][0, 0, 0, %[[VAL_129]]] : (!llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>, i32) -> !llvm.ptr<i64>
    // CHECK:             %[[VAL_133:.*]] = llvm.getelementptr %[[VAL_132]]{{\[}}%[[VAL_118]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    // CHECK:             llvm.store %[[VAL_131]], %[[VAL_133]] : !llvm.ptr<i64>
    // CHECK:             %[[VAL_134:.*]] = llvm.getelementptr %[[VAL_117]]{{\[}}%[[VAL_118]]] : (!llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>, i64) -> !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
    // CHECK:             %[[VAL_135:.*]] = llvm.load %[[VAL_134]] : !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
    // CHECK:             llvm.return %[[VAL_135]] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
    // CHECK:           }
    func.func @test_work_group_size() -> !sycl_range_3_ {
      %0 = sycl.work_group_size : !sycl_range_3_
      return %0 : !sycl_range_3_
    }

    // CHECK-LABEL:     llvm.func @test_work_group_size_dim(
    // CHECK-SAME:                                          %[[VAL_136:.*]]: i32) -> i64 {
    // CHECK:             %[[VAL_137:.*]] = llvm.mlir.addressof @__builtin_var_WorkgroupSize__ : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_138:.*]] = llvm.load %[[VAL_137]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_139:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
    // CHECK-DAG:         %[[VAL_140:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:             %[[VAL_141:.*]] = llvm.extractelement %[[VAL_138]]{{\[}}%[[VAL_140]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_142:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK:             %[[VAL_143:.*]] = llvm.insertelement %[[VAL_141]], %[[VAL_139]]{{\[}}%[[VAL_142]] : i64] : vector<3xi64>
    // CHECK:             %[[VAL_144:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:             %[[VAL_145:.*]] = llvm.extractelement %[[VAL_138]]{{\[}}%[[VAL_144]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_146:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK:             %[[VAL_147:.*]] = llvm.insertelement %[[VAL_145]], %[[VAL_143]]{{\[}}%[[VAL_146]] : i64] : vector<3xi64>
    // CHECK:             %[[VAL_148:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:             %[[VAL_149:.*]] = llvm.extractelement %[[VAL_138]]{{\[}}%[[VAL_148]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_150:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:             %[[VAL_151:.*]] = llvm.insertelement %[[VAL_149]], %[[VAL_147]]{{\[}}%[[VAL_150]] : i64] : vector<3xi64>
    // CHECK:             %[[VAL_152:.*]] = llvm.extractelement %[[VAL_151]]{{\[}}%[[VAL_136]] : i32] : vector<3xi64>
    // CHECK:             llvm.return %[[VAL_152]] : i64
    // CHECK:           }
    func.func @test_work_group_size_dim(%i: i32) -> index {
      %0 = sycl.work_group_size %i : index
      return %0 : index
    }

    // CHECK-LABEL:     llvm.func @test_work_group_id() -> !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> {
    // CHECK:             %[[VAL_153:.*]] = llvm.mlir.addressof @__builtin_var_WorkgroupId__ : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_154:.*]] = llvm.load %[[VAL_153]] : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_155:.*]] = llvm.mlir.null : !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK:             %[[VAL_156:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK:             %[[VAL_157:.*]] = llvm.getelementptr %[[VAL_155]]{{\[}}%[[VAL_156]]] : (!llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>, i64) -> !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK:             %[[VAL_158:.*]] = llvm.ptrtoint %[[VAL_157]] : !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>> to i64
    // CHECK:             %[[VAL_159:.*]] = llvm.alloca %[[VAL_158]] x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> : (i64) -> !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_160:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_161:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:         %[[VAL_162:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:             %[[VAL_163:.*]] = llvm.extractelement %[[VAL_154]]{{\[}}%[[VAL_162]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_164:.*]] = llvm.getelementptr inbounds %[[VAL_159]][0, 0, 0, %[[VAL_161]]] : (!llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>, i32) -> !llvm.ptr<i64>
    // CHECK:             %[[VAL_165:.*]] = llvm.getelementptr %[[VAL_164]]{{\[}}%[[VAL_160]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    // CHECK:             llvm.store %[[VAL_163]], %[[VAL_165]] : !llvm.ptr<i64>
    // CHECK:             %[[VAL_166:.*]] = llvm.getelementptr %[[VAL_159]]{{\[}}%[[VAL_160]]] : (!llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>, i64) -> !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK:             %[[VAL_167:.*]] = llvm.load %[[VAL_166]] : !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK:             llvm.return %[[VAL_167]] : !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>
    // CHECK:           }
    func.func @test_work_group_id() -> !sycl_id_1_ {
      %0 = sycl.work_group_id : !sycl_id_1_
      return %0 : !sycl_id_1_
    }

    // CHECK-LABEL:     llvm.func @test_work_group_id_dim(
    // CHECK-SAME:                                        %[[VAL_168:.*]]: i32) -> i64 {
    // CHECK:             %[[VAL_169:.*]] = llvm.mlir.addressof @__builtin_var_WorkgroupId__ : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_170:.*]] = llvm.load %[[VAL_169]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_171:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
    // CHECK-DAG:         %[[VAL_172:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:             %[[VAL_173:.*]] = llvm.extractelement %[[VAL_170]]{{\[}}%[[VAL_172]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_174:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK:             %[[VAL_175:.*]] = llvm.insertelement %[[VAL_173]], %[[VAL_171]]{{\[}}%[[VAL_174]] : i64] : vector<3xi64>
    // CHECK:             %[[VAL_176:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:             %[[VAL_177:.*]] = llvm.extractelement %[[VAL_170]]{{\[}}%[[VAL_176]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_178:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK:             %[[VAL_179:.*]] = llvm.insertelement %[[VAL_177]], %[[VAL_175]]{{\[}}%[[VAL_178]] : i64] : vector<3xi64>
    // CHECK:             %[[VAL_180:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:             %[[VAL_181:.*]] = llvm.extractelement %[[VAL_170]]{{\[}}%[[VAL_180]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_182:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:             %[[VAL_183:.*]] = llvm.insertelement %[[VAL_181]], %[[VAL_179]]{{\[}}%[[VAL_182]] : i64] : vector<3xi64>
    // CHECK:             %[[VAL_184:.*]] = llvm.extractelement %[[VAL_183]]{{\[}}%[[VAL_168]] : i32] : vector<3xi64>
    // CHECK:             llvm.return %[[VAL_184]] : i64
    // CHECK:           }
    func.func @test_work_group_id_dim(%i: i32) -> index {
      %0 = sycl.work_group_id %i : index
      return %0 : index
    }

    // CHECK-LABEL:     llvm.func @test_num_sub_groups() -> i32 {
    // CHECK-NEXT:        %[[VAL_200:.*]] = llvm.mlir.addressof @__builtin_var_NumSubgroups__ : !llvm.ptr<i64>
    // CHECK-NEXT:        %[[VAL_201:.*]] = llvm.load %[[VAL_200]] : !llvm.ptr<i64>
    // CHECK-NEXT:        %[[VAL_202:.*]] = llvm.trunc %[[VAL_201]] : i64 to i32
    // CHECK-NEXT:        llvm.return %[[VAL_202]] : i32
    // CHECK-NEXT:      }
    func.func @test_num_sub_groups() -> i32 {
      %0 = sycl.num_sub_groups : i32
      return %0 : i32
    }

    // CHECK-LABEL:     llvm.func @test_sub_group_size() -> i32 {
    // CHECK-NEXT:        %[[VAL_203:.*]] = llvm.mlir.addressof @__builtin_var_SubgroupSize__ : !llvm.ptr<i64>
    // CHECK-NEXT:        %[[VAL_204:.*]] = llvm.load %[[VAL_203]] : !llvm.ptr<i64>
    // CHECK-NEXT:        %[[VAL_205:.*]] = llvm.trunc %[[VAL_204]] : i64 to i32
    // CHECK-NEXT:        llvm.return %[[VAL_205]] : i32
    // CHECK-NEXT:      }
    func.func @test_sub_group_size() -> i32 {
      %0 = sycl.sub_group_size : i32
      return %0 : i32
    }

    // CHECK-LABEL:     llvm.func @test_sub_group_id() -> i32 {
    // CHECK-NEXT:        %[[VAL_206:.*]] = llvm.mlir.addressof @__builtin_var_SubgroupId__ : !llvm.ptr<i64>
    // CHECK-NEXT:        %[[VAL_207:.*]] = llvm.load %[[VAL_206]] : !llvm.ptr<i64>
    // CHECK-NEXT:        %[[VAL_208:.*]] = llvm.trunc %[[VAL_207]] : i64 to i32
    // CHECK-NEXT:        llvm.return %[[VAL_208]] : i32
    // CHECK-NEXT:      }
    func.func @test_sub_group_id() -> i32 {
      %0 = sycl.sub_group_id : i32
      return %0 : i32
    }

    // CHECK-LABEL:     llvm.func @test_global_offset() -> !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> {
    // CHECK:             %[[VAL_194:.*]] = llvm.mlir.addressof @__builtin_var_GlobalOffset__ : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_195:.*]] = llvm.load %[[VAL_194]] : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_196:.*]] = llvm.mlir.null : !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK:             %[[VAL_197:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK:             %[[VAL_198:.*]] = llvm.getelementptr %[[VAL_196]]{{\[}}%[[VAL_197]]] : (!llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>, i64) -> !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK:             %[[VAL_199:.*]] = llvm.ptrtoint %[[VAL_198]] : !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>> to i64
    // CHECK:             %[[VAL_200:.*]] = llvm.alloca %[[VAL_199]] x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> : (i64) -> !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_201:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_202:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:         %[[VAL_203:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:             %[[VAL_204:.*]] = llvm.extractelement %[[VAL_195]]{{\[}}%[[VAL_203]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_205:.*]] = llvm.getelementptr inbounds %[[VAL_200]][0, 0, 0, %[[VAL_202]]] : (!llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>, i32) -> !llvm.ptr<i64>
    // CHECK:             %[[VAL_206:.*]] = llvm.getelementptr %[[VAL_205]]{{\[}}%[[VAL_201]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    // CHECK:             llvm.store %[[VAL_204]], %[[VAL_206]] : !llvm.ptr<i64>
    // CHECK:             %[[VAL_207:.*]] = llvm.getelementptr %[[VAL_200]]{{\[}}%[[VAL_201]]] : (!llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>, i64) -> !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK:             %[[VAL_208:.*]] = llvm.load %[[VAL_207]] : !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK:             llvm.return %[[VAL_208]] : !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>
    // CHECK:           }
    func.func @test_global_offset() -> !sycl_id_1_ {
      %0 = sycl.global_offset : !sycl_id_1_
      return %0 : !sycl_id_1_
    }

    // CHECK-LABEL:     llvm.func @test_global_offset_dim(
    // CHECK-SAME:                                        %[[VAL_222:.*]]: i32) -> i64 {
    // CHECK-NEXT:        %[[VAL_223:.*]] = llvm.mlir.addressof @__builtin_var_GlobalOffset__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_224:.*]] = llvm.load %[[VAL_223]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_225:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
    // CHECK-DAG:         %[[VAL_226:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_227:.*]] = llvm.extractelement %[[VAL_224]]{{\[}}%[[VAL_226]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_228:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT:        %[[VAL_229:.*]] = llvm.insertelement %[[VAL_227]], %[[VAL_225]]{{\[}}%[[VAL_228]] : i64] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_230:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:        %[[VAL_231:.*]] = llvm.extractelement %[[VAL_224]]{{\[}}%[[VAL_230]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_232:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-NEXT:        %[[VAL_233:.*]] = llvm.insertelement %[[VAL_231]], %[[VAL_229]]{{\[}}%[[VAL_232]] : i64] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_234:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT:        %[[VAL_235:.*]] = llvm.extractelement %[[VAL_224]]{{\[}}%[[VAL_234]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_236:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK-NEXT:        %[[VAL_237:.*]] = llvm.insertelement %[[VAL_235]], %[[VAL_233]]{{\[}}%[[VAL_236]] : i64] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_238:.*]] = llvm.extractelement %[[VAL_237]]{{\[}}%[[VAL_222]] : i32] : vector<3xi64>
    // CHECK-NEXT:        llvm.return %[[VAL_238]] : i64
    // CHECK-NEXT:      }
    func.func @test_global_offset_dim(%i: i32) -> index {
      %0 = sycl.global_offset %i : index
      return %0 : index
    }

    // CHECK-LABEL:     llvm.func @test_num_work_groups() -> !llvm.struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)> {
    // CHECK:             %[[VAL_226:.*]] = llvm.mlir.addressof @__builtin_var_NumWorkgroups__ : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_227:.*]] = llvm.load %[[VAL_226]] : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_228:.*]] = llvm.mlir.null : !llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>
    // CHECK:             %[[VAL_229:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK:             %[[VAL_230:.*]] = llvm.getelementptr %[[VAL_228]]{{\[}}%[[VAL_229]]] : (!llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>, i64) -> !llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>
    // CHECK:             %[[VAL_231:.*]] = llvm.ptrtoint %[[VAL_230]] : !llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>> to i64
    // CHECK:             %[[VAL_232:.*]] = llvm.alloca %[[VAL_231]] x !llvm.struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)> : (i64) -> !llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_233:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_234:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:         %[[VAL_235:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:             %[[VAL_236:.*]] = llvm.extractelement %[[VAL_227]]{{\[}}%[[VAL_235]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_237:.*]] = llvm.getelementptr inbounds %[[VAL_232]][0, 0, 0, %[[VAL_234]]] : (!llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>, i32) -> !llvm.ptr<i64>
    // CHECK:             %[[VAL_238:.*]] = llvm.getelementptr %[[VAL_237]]{{\[}}%[[VAL_233]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    // CHECK:             llvm.store %[[VAL_236]], %[[VAL_238]] : !llvm.ptr<i64>
    // CHECK-DAG:         %[[VAL_239:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:         %[[VAL_240:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:             %[[VAL_241:.*]] = llvm.extractelement %[[VAL_227]]{{\[}}%[[VAL_240]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_242:.*]] = llvm.getelementptr inbounds %[[VAL_232]][0, 0, 0, %[[VAL_239]]] : (!llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>, i32) -> !llvm.ptr<i64>
    // CHECK:             %[[VAL_243:.*]] = llvm.getelementptr %[[VAL_242]]{{\[}}%[[VAL_233]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
    // CHECK:             llvm.store %[[VAL_241]], %[[VAL_243]] : !llvm.ptr<i64>
    // CHECK:             %[[VAL_244:.*]] = llvm.getelementptr %[[VAL_232]]{{\[}}%[[VAL_233]]] : (!llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>, i64) -> !llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>
    // CHECK:             %[[VAL_245:.*]] = llvm.load %[[VAL_244]] : !llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>
    // CHECK:             llvm.return %[[VAL_245]] : !llvm.struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>
    // CHECK:           }
    func.func @test_num_work_groups() -> !sycl_range_2_ {
      %0 = sycl.num_work_groups : !sycl_range_2_
      return %0 : !sycl_range_2_
    }

    // CHECK-LABEL:     llvm.func @test_num_work_groups_dim(
    // CHECK-SAME:                                          %[[VAL_246:.*]]: i32) -> i64 {
    // CHECK:             %[[VAL_247:.*]] = llvm.mlir.addressof @__builtin_var_NumWorkgroups__ : !llvm.ptr<vector<3xi64>>
    // CHECK:             %[[VAL_248:.*]] = llvm.load %[[VAL_247]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_249:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
    // CHECK-DAG:         %[[VAL_250:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:             %[[VAL_251:.*]] = llvm.extractelement %[[VAL_248]]{{\[}}%[[VAL_250]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_252:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK:             %[[VAL_253:.*]] = llvm.insertelement %[[VAL_251]], %[[VAL_249]]{{\[}}%[[VAL_252]] : i64] : vector<3xi64>
    // CHECK:             %[[VAL_254:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:             %[[VAL_255:.*]] = llvm.extractelement %[[VAL_248]]{{\[}}%[[VAL_254]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_256:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK:             %[[VAL_257:.*]] = llvm.insertelement %[[VAL_255]], %[[VAL_253]]{{\[}}%[[VAL_256]] : i64] : vector<3xi64>
    // CHECK:             %[[VAL_258:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:             %[[VAL_259:.*]] = llvm.extractelement %[[VAL_248]]{{\[}}%[[VAL_258]] : i32] : vector<3xi64>
    // CHECK:             %[[VAL_260:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:             %[[VAL_261:.*]] = llvm.insertelement %[[VAL_259]], %[[VAL_257]]{{\[}}%[[VAL_260]] : i64] : vector<3xi64>
    // CHECK:             %[[VAL_262:.*]] = llvm.extractelement %[[VAL_261]]{{\[}}%[[VAL_246]] : i32] : vector<3xi64>
    // CHECK:             llvm.return %[[VAL_262]] : i64
    // CHECK:           }
    func.func @test_num_work_groups_dim(%i: i32) -> index {
      %0 = sycl.num_work_groups %i : index
      return %0 : index
    }

    // CHECK-LABEL:     llvm.func @test_sub_group_max_size() -> i32 {
    // CHECK-NEXT:        %[[VAL_273:.*]] = llvm.mlir.addressof @__builtin_var_SubgroupMaxSize__ : !llvm.ptr<i64>
    // CHECK-NEXT:        %[[VAL_274:.*]] = llvm.load %[[VAL_273]] : !llvm.ptr<i64>
    // CHECK-NEXT:        %[[VAL_275:.*]] = llvm.trunc %[[VAL_274]] : i64 to i32
    // CHECK-NEXT:        llvm.return %[[VAL_275]] : i32
    func.func @test_sub_group_max_size() -> i32 {
      %0 = sycl.sub_group_max_size : i32
      return %0 : i32
    }

    // CHECK-LABEL:     llvm.func @test_sub_group_local_id() -> i32 {
    // CHECK-NEXT:        %[[VAL_276:.*]] = llvm.mlir.addressof @__builtin_var_SubgroupLocalInvocationId__ : !llvm.ptr<i64>
    // CHECK-NEXT:        %[[VAL_277:.*]] = llvm.load %[[VAL_276]] : !llvm.ptr<i64>
    // CHECK-NEXT:        %[[VAL_278:.*]] = llvm.trunc %[[VAL_277]] : i64 to i32
    // CHECK-NEXT:        llvm.return %[[VAL_278]] : i32
    // CHECK-NEXT:      }
    func.func @test_sub_group_local_id() -> i32 {
      %0 = sycl.sub_group_local_id : i32
      return %0 : i32
    }
  }
}
