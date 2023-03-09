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
  gpu.module @kernels {
    // CHECK-LABEL:   llvm.func @test_num_work_items() -> !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> {
    // CHECK-DAG:         %[[VAL_0:.*]] = llvm.mlir.null : !llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_1:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_2:.*]] = llvm.getelementptr %[[VAL_0]][1] : (!llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>) -> !llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK-NEXT:        %[[VAL_3:.*]] = llvm.ptrtoint %[[VAL_2]] : !llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>> to i64
    // CHECK-NEXT:        %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> : (i64) -> !llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_5:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_6:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_7:.*]] = llvm.mlir.addressof @__builtin_var_NumWorkgroups__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_8:.*]] = llvm.load %[[VAL_7]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_9:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_10:.*]] = llvm.extractelement %[[VAL_8]]{{\[}}%[[VAL_9]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_4]][0, 0, 0, 0] : (!llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>) -> !llvm.ptr<i64, 4>
    // CHECK-NEXT:        llvm.store %[[VAL_10]], %[[VAL_11]] : !llvm.ptr<i64, 4>
    // CHECK-NEXT:        %[[VAL_12:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK-NEXT:        llvm.return %[[VAL_12]] : !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>
    // CHECK-NEXT:      }
    func.func @test_num_work_items() -> !sycl_range_1_ {
      %0 = sycl.num_work_items : !sycl_range_1_
      return %0 : !sycl_range_1_
    }

    // CHECK-LABEL:     llvm.func @test_num_work_items_dim(
    // CHECK-SAME:                                         %[[VAL_13:.*]]: i32) -> i64 {
    // CHECK-DAG:         %[[VAL_14:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_15:.*]] = llvm.mlir.addressof @__builtin_var_NumWorkgroups__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_16:.*]] = llvm.load %[[VAL_15]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_17:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_18:.*]] = llvm.extractelement %[[VAL_16]]{{\[}}%[[VAL_17]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_19:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT:        %[[VAL_20:.*]] = llvm.insertelement %[[VAL_18]], %[[VAL_14]]{{\[}}%[[VAL_19]] : i64] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_21:.*]] = llvm.mlir.addressof @__builtin_var_NumWorkgroups__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_22:.*]] = llvm.load %[[VAL_21]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_23:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:        %[[VAL_24:.*]] = llvm.extractelement %[[VAL_22]]{{\[}}%[[VAL_23]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_25:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-NEXT:        %[[VAL_26:.*]] = llvm.insertelement %[[VAL_24]], %[[VAL_20]]{{\[}}%[[VAL_25]] : i64] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_27:.*]] = llvm.mlir.addressof @__builtin_var_NumWorkgroups__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_28:.*]] = llvm.load %[[VAL_27]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_29:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT:        %[[VAL_30:.*]] = llvm.extractelement %[[VAL_28]]{{\[}}%[[VAL_29]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_31:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK-NEXT:        %[[VAL_32:.*]] = llvm.insertelement %[[VAL_30]], %[[VAL_26]]{{\[}}%[[VAL_31]] : i64] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_33:.*]] = llvm.extractelement %[[VAL_32]]{{\[}}%[[VAL_13]] : i32] : vector<3xi64>
    // CHECK-NEXT:        llvm.return %[[VAL_33]] : i64
    // CHECK-NEXT:      }
    func.func @test_num_work_items_dim(%i: i32) -> index {
      %0 = sycl.num_work_items %i : index
      return %0 : index
    }

    // CHECK-LABEL:     llvm.func @test_global_id() -> !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)> {
    // CHECK-DAG:         %[[VAL_34:.*]] = llvm.mlir.null : !llvm.ptr<struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_35:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_36:.*]] = llvm.getelementptr %[[VAL_34]][1] : (!llvm.ptr<struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>) -> !llvm.ptr<struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>
    // CHECK-NEXT:        %[[VAL_37:.*]] = llvm.ptrtoint %[[VAL_36]] : !llvm.ptr<struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>> to i64
    // CHECK-NEXT:        %[[VAL_38:.*]] = llvm.alloca %[[VAL_37]] x !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)> : (i64) -> !llvm.ptr<struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_39:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_40:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_41:.*]] = llvm.mlir.addressof @__builtin_var_GlobalInvocationId__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_42:.*]] = llvm.load %[[VAL_41]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_43:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_44:.*]] = llvm.extractelement %[[VAL_42]]{{\[}}%[[VAL_43]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_45:.*]] = llvm.getelementptr inbounds %[[VAL_38]][0, 0, 0, 0] : (!llvm.ptr<struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>) -> !llvm.ptr<i64, 4>
    // CHECK-NEXT:        llvm.store %[[VAL_44]], %[[VAL_45]] : !llvm.ptr<i64, 4>
    // CHECK-DAG:         %[[VAL_46:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:        %[[VAL_47:.*]] = llvm.mlir.addressof @__builtin_var_GlobalInvocationId__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_48:.*]] = llvm.load %[[VAL_47]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_49:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:        %[[VAL_50:.*]] = llvm.extractelement %[[VAL_48]]{{\[}}%[[VAL_49]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_51:.*]] = llvm.getelementptr inbounds %[[VAL_38]][0, 0, 0, 1] : (!llvm.ptr<struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>) -> !llvm.ptr<i64, 4>
    // CHECK-NEXT:        llvm.store %[[VAL_50]], %[[VAL_51]] : !llvm.ptr<i64, 4>
    // CHECK-NEXT:        %[[VAL_52:.*]] = llvm.load %[[VAL_38]] : !llvm.ptr<struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>
    // CHECK-NEXT:        llvm.return %[[VAL_52]] : !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>
    // CHECK-NEXT:      }
    func.func @test_global_id() -> !sycl_id_2_ {
      %0 = sycl.global_id : !sycl_id_2_
      return %0 : !sycl_id_2_
    }

    // CHECK-LABEL:     llvm.func @test_global_id_dim(
    // CHECK-SAME:                                    %[[VAL_53:.*]]: i32) -> i64 {
    // CHECK-DAG:         %[[VAL_54:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_55:.*]] = llvm.mlir.addressof @__builtin_var_GlobalInvocationId__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_56:.*]] = llvm.load %[[VAL_55]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_57:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_58:.*]] = llvm.extractelement %[[VAL_56]]{{\[}}%[[VAL_57]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_59:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT:        %[[VAL_60:.*]] = llvm.insertelement %[[VAL_58]], %[[VAL_54]]{{\[}}%[[VAL_59]] : i64] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_61:.*]] = llvm.mlir.addressof @__builtin_var_GlobalInvocationId__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_62:.*]] = llvm.load %[[VAL_61]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_63:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:        %[[VAL_64:.*]] = llvm.extractelement %[[VAL_62]]{{\[}}%[[VAL_63]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_65:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-NEXT:        %[[VAL_66:.*]] = llvm.insertelement %[[VAL_64]], %[[VAL_60]]{{\[}}%[[VAL_65]] : i64] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_67:.*]] = llvm.mlir.addressof @__builtin_var_GlobalInvocationId__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_68:.*]] = llvm.load %[[VAL_67]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_69:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT:        %[[VAL_70:.*]] = llvm.extractelement %[[VAL_68]]{{\[}}%[[VAL_69]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_71:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK-NEXT:        %[[VAL_72:.*]] = llvm.insertelement %[[VAL_70]], %[[VAL_66]]{{\[}}%[[VAL_71]] : i64] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_73:.*]] = llvm.extractelement %[[VAL_72]]{{\[}}%[[VAL_53]] : i32] : vector<3xi64>
    // CHECK-NEXT:        llvm.return %[[VAL_73]] : i64
    // CHECK-NEXT:      }
    func.func @test_global_id_dim(%i: i32) -> index {
      %0 = sycl.global_id %i : index
      return %0 : index
    }

    // CHECK-LABEL:     llvm.func @test_local_id() -> !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {
    // CHECK-DAG:         %[[VAL_74:.*]] = llvm.mlir.null : !llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_75:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_76:.*]] = llvm.getelementptr %[[VAL_74]][1] : (!llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>) -> !llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
    // CHECK-NEXT:        %[[VAL_77:.*]] = llvm.ptrtoint %[[VAL_76]] : !llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>> to i64
    // CHECK-NEXT:        %[[VAL_78:.*]] = llvm.alloca %[[VAL_77]] x !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> : (i64) -> !llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_79:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_80:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_81:.*]] = llvm.mlir.addressof @__builtin_var_LocalInvocationId__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_82:.*]] = llvm.load %[[VAL_81]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_83:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_84:.*]] = llvm.extractelement %[[VAL_82]]{{\[}}%[[VAL_83]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_85:.*]] = llvm.getelementptr inbounds %[[VAL_78]][0, 0, 0, 0] : (!llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>) -> !llvm.ptr<i64, 4>
    // CHECK-NEXT:        llvm.store %[[VAL_84]], %[[VAL_85]] : !llvm.ptr<i64, 4>
    // CHECK-DAG:         %[[VAL_86:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:        %[[VAL_87:.*]] = llvm.mlir.addressof @__builtin_var_LocalInvocationId__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_88:.*]] = llvm.load %[[VAL_87]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_89:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:        %[[VAL_90:.*]] = llvm.extractelement %[[VAL_88]]{{\[}}%[[VAL_89]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_91:.*]] = llvm.getelementptr inbounds %[[VAL_78]][0, 0, 0, 1] : (!llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>) -> !llvm.ptr<i64, 4>
    // CHECK-NEXT:        llvm.store %[[VAL_90]], %[[VAL_91]] : !llvm.ptr<i64, 4>
    // CHECK-DAG:         %[[VAL_92:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT:        %[[VAL_93:.*]] = llvm.mlir.addressof @__builtin_var_LocalInvocationId__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_94:.*]] = llvm.load %[[VAL_93]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_95:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT:        %[[VAL_96:.*]] = llvm.extractelement %[[VAL_94]]{{\[}}%[[VAL_95]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_97:.*]] = llvm.getelementptr inbounds %[[VAL_78]][0, 0, 0, 2] : (!llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>) -> !llvm.ptr<i64, 4>
    // CHECK-NEXT:        llvm.store %[[VAL_96]], %[[VAL_97]] : !llvm.ptr<i64, 4>
    // CHECK-NEXT:        %[[VAL_98:.*]] = llvm.load %[[VAL_78]] : !llvm.ptr<struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
    // CHECK-NEXT:        llvm.return %[[VAL_98]] : !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
    // CHECK-NEXT:      }
    func.func @test_local_id() -> !sycl_id_3_ {
      %0 = sycl.local_id : !sycl_id_3_
      return %0 : !sycl_id_3_
    }

    // CHECK-LABEL:     llvm.func @test_local_id_dim(
    // CHECK-SAME:                                   %[[VAL_99:.*]]: i32) -> i64 {
    // CHECK-DAG:         %[[VAL_100:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_101:.*]] = llvm.mlir.addressof @__builtin_var_LocalInvocationId__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_102:.*]] = llvm.load %[[VAL_101]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_103:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_104:.*]] = llvm.extractelement %[[VAL_102]]{{\[}}%[[VAL_103]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_105:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT:        %[[VAL_106:.*]] = llvm.insertelement %[[VAL_104]], %[[VAL_100]]{{\[}}%[[VAL_105]] : i64] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_107:.*]] = llvm.mlir.addressof @__builtin_var_LocalInvocationId__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_108:.*]] = llvm.load %[[VAL_107]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_109:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:        %[[VAL_110:.*]] = llvm.extractelement %[[VAL_108]]{{\[}}%[[VAL_109]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_111:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-NEXT:        %[[VAL_112:.*]] = llvm.insertelement %[[VAL_110]], %[[VAL_106]]{{\[}}%[[VAL_111]] : i64] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_113:.*]] = llvm.mlir.addressof @__builtin_var_LocalInvocationId__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_114:.*]] = llvm.load %[[VAL_113]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_115:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT:        %[[VAL_116:.*]] = llvm.extractelement %[[VAL_114]]{{\[}}%[[VAL_115]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_117:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK-NEXT:        %[[VAL_118:.*]] = llvm.insertelement %[[VAL_116]], %[[VAL_112]]{{\[}}%[[VAL_117]] : i64] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_119:.*]] = llvm.extractelement %[[VAL_118]]{{\[}}%[[VAL_99]] : i32] : vector<3xi64>
    // CHECK-NEXT:        llvm.return %[[VAL_119]] : i64
    // CHECK-NEXT:      }
    func.func @test_local_id_dim(%i: i32) -> index {
      %0 = sycl.local_id %i : index
      return %0 : index
    }

    // CHECK-LABEL:     llvm.func @test_work_group_size() -> !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {
    // CHECK-DAG:         %[[VAL_120:.*]] = llvm.mlir.null : !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_121:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_122:.*]] = llvm.getelementptr %[[VAL_120]][1] : (!llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>) -> !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
    // CHECK-NEXT:        %[[VAL_123:.*]] = llvm.ptrtoint %[[VAL_122]] : !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>> to i64
    // CHECK-NEXT:        %[[VAL_124:.*]] = llvm.alloca %[[VAL_123]] x !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> : (i64) -> !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_125:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_126:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_127:.*]] = llvm.mlir.addressof @__builtin_var_WorkgroupSize__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_128:.*]] = llvm.load %[[VAL_127]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_129:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_130:.*]] = llvm.extractelement %[[VAL_128]]{{\[}}%[[VAL_129]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_131:.*]] = llvm.getelementptr inbounds %[[VAL_124]][0, 0, 0, 0] : (!llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>) -> !llvm.ptr<i64, 4>
    // CHECK-NEXT:        llvm.store %[[VAL_130]], %[[VAL_131]] : !llvm.ptr<i64, 4>
    // CHECK-DAG:         %[[VAL_132:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:        %[[VAL_133:.*]] = llvm.mlir.addressof @__builtin_var_WorkgroupSize__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_134:.*]] = llvm.load %[[VAL_133]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_135:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:        %[[VAL_136:.*]] = llvm.extractelement %[[VAL_134]]{{\[}}%[[VAL_135]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_137:.*]] = llvm.getelementptr inbounds %[[VAL_124]][0, 0, 0, 1] : (!llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>) -> !llvm.ptr<i64, 4>
    // CHECK-NEXT:        llvm.store %[[VAL_136]], %[[VAL_137]] : !llvm.ptr<i64, 4>
    // CHECK-DAG:         %[[VAL_138:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT:        %[[VAL_139:.*]] = llvm.mlir.addressof @__builtin_var_WorkgroupSize__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_140:.*]] = llvm.load %[[VAL_139]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_141:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT:        %[[VAL_142:.*]] = llvm.extractelement %[[VAL_140]]{{\[}}%[[VAL_141]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_143:.*]] = llvm.getelementptr inbounds %[[VAL_124]][0, 0, 0, 2] : (!llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>) -> !llvm.ptr<i64, 4>
    // CHECK-NEXT:        llvm.store %[[VAL_142]], %[[VAL_143]] : !llvm.ptr<i64, 4>
    // CHECK-NEXT:        %[[VAL_144:.*]] = llvm.load %[[VAL_124]] : !llvm.ptr<struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>>
    // CHECK-NEXT:        llvm.return %[[VAL_144]] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
    // CHECK-NEXT:      }
    func.func @test_work_group_size() -> !sycl_range_3_ {
      %0 = sycl.work_group_size : !sycl_range_3_
      return %0 : !sycl_range_3_
    }

    // CHECK-LABEL:     llvm.func @test_work_group_size_dim(
    // CHECK-SAME:                                          %[[VAL_145:.*]]: i32) -> i64 {
    // CHECK-DAG:         %[[VAL_146:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_147:.*]] = llvm.mlir.addressof @__builtin_var_WorkgroupSize__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_148:.*]] = llvm.load %[[VAL_147]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_149:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_150:.*]] = llvm.extractelement %[[VAL_148]]{{\[}}%[[VAL_149]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_151:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT:        %[[VAL_152:.*]] = llvm.insertelement %[[VAL_150]], %[[VAL_146]]{{\[}}%[[VAL_151]] : i64] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_153:.*]] = llvm.mlir.addressof @__builtin_var_WorkgroupSize__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_154:.*]] = llvm.load %[[VAL_153]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_155:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:        %[[VAL_156:.*]] = llvm.extractelement %[[VAL_154]]{{\[}}%[[VAL_155]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_157:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-NEXT:        %[[VAL_158:.*]] = llvm.insertelement %[[VAL_156]], %[[VAL_152]]{{\[}}%[[VAL_157]] : i64] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_159:.*]] = llvm.mlir.addressof @__builtin_var_WorkgroupSize__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_160:.*]] = llvm.load %[[VAL_159]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_161:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT:        %[[VAL_162:.*]] = llvm.extractelement %[[VAL_160]]{{\[}}%[[VAL_161]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_163:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK-NEXT:        %[[VAL_164:.*]] = llvm.insertelement %[[VAL_162]], %[[VAL_158]]{{\[}}%[[VAL_163]] : i64] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_165:.*]] = llvm.extractelement %[[VAL_164]]{{\[}}%[[VAL_145]] : i32] : vector<3xi64>
    // CHECK-NEXT:        llvm.return %[[VAL_165]] : i64
    // CHECK-NEXT:      }
    func.func @test_work_group_size_dim(%i: i32) -> index {
      %0 = sycl.work_group_size %i : index
      return %0 : index
    }

    // CHECK-LABEL:     llvm.func @test_work_group_id() -> !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> {
    // CHECK-DAG:         %[[VAL_166:.*]] = llvm.mlir.null : !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_167:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_168:.*]] = llvm.getelementptr %[[VAL_166]][1] : (!llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>) -> !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK-NEXT:        %[[VAL_169:.*]] = llvm.ptrtoint %[[VAL_168]] : !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>> to i64
    // CHECK-NEXT:        %[[VAL_170:.*]] = llvm.alloca %[[VAL_169]] x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> : (i64) -> !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_171:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_172:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_173:.*]] = llvm.mlir.addressof @__builtin_var_WorkgroupId__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_174:.*]] = llvm.load %[[VAL_173]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_175:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_176:.*]] = llvm.extractelement %[[VAL_174]]{{\[}}%[[VAL_175]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_177:.*]] = llvm.getelementptr inbounds %[[VAL_170]][0, 0, 0, 0] : (!llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>) -> !llvm.ptr<i64, 4>
    // CHECK-NEXT:        llvm.store %[[VAL_176]], %[[VAL_177]] : !llvm.ptr<i64, 4>
    // CHECK-NEXT:        %[[VAL_178:.*]] = llvm.load %[[VAL_170]] : !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK-NEXT:        llvm.return %[[VAL_178]] : !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>
    // CHECK-NEXT:      }
    func.func @test_work_group_id() -> !sycl_id_1_ {
      %0 = sycl.work_group_id : !sycl_id_1_
      return %0 : !sycl_id_1_
    }

    // CHECK-LABEL:     llvm.func @test_work_group_id_dim(
    // CHECK-SAME:                                        %[[VAL_179:.*]]: i32) -> i64 {
    // CHECK-DAG:         %[[VAL_180:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_181:.*]] = llvm.mlir.addressof @__builtin_var_WorkgroupId__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_182:.*]] = llvm.load %[[VAL_181]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_183:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_184:.*]] = llvm.extractelement %[[VAL_182]]{{\[}}%[[VAL_183]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_185:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT:        %[[VAL_186:.*]] = llvm.insertelement %[[VAL_184]], %[[VAL_180]]{{\[}}%[[VAL_185]] : i64] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_187:.*]] = llvm.mlir.addressof @__builtin_var_WorkgroupId__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_188:.*]] = llvm.load %[[VAL_187]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_189:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:        %[[VAL_190:.*]] = llvm.extractelement %[[VAL_188]]{{\[}}%[[VAL_189]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_191:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-NEXT:        %[[VAL_192:.*]] = llvm.insertelement %[[VAL_190]], %[[VAL_186]]{{\[}}%[[VAL_191]] : i64] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_193:.*]] = llvm.mlir.addressof @__builtin_var_WorkgroupId__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_194:.*]] = llvm.load %[[VAL_193]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_195:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT:        %[[VAL_196:.*]] = llvm.extractelement %[[VAL_194]]{{\[}}%[[VAL_195]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_197:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK-NEXT:        %[[VAL_198:.*]] = llvm.insertelement %[[VAL_196]], %[[VAL_192]]{{\[}}%[[VAL_197]] : i64] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_199:.*]] = llvm.extractelement %[[VAL_198]]{{\[}}%[[VAL_179]] : i32] : vector<3xi64>
    // CHECK-NEXT:        llvm.return %[[VAL_199]] : i64
    // CHECK-NEXT:      }
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
    // CHECK-NEXT:        %[[VAL_209:.*]] = llvm.mlir.addressof @__builtin_var_GlobalOffset__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_210:.*]] = llvm.load %[[VAL_209]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_211:.*]] = llvm.mlir.null : !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_212:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_213:.*]] = llvm.getelementptr %[[VAL_211]][1] : (!llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>) -> !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK-NEXT:        %[[VAL_214:.*]] = llvm.ptrtoint %[[VAL_213]] : !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>> to i64
    // CHECK-NEXT:        %[[VAL_215:.*]] = llvm.alloca %[[VAL_214]] x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> : (i64) -> !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_216:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_217:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:         %[[VAL_218:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_219:.*]] = llvm.extractelement %[[VAL_210]]{{\[}}%[[VAL_218]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_220:.*]] = llvm.getelementptr inbounds %[[VAL_215]][0, 0, 0, 0] : (!llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>) -> !llvm.ptr<i64, 4>
    // CHECK-NEXT:        llvm.store %[[VAL_219]], %[[VAL_220]] : !llvm.ptr<i64, 4>
    // CHECK-NEXT:        %[[VAL_221:.*]] = llvm.load %[[VAL_215]] : !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>
    // CHECK-NEXT:        llvm.return %[[VAL_221]] : !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>
    // CHECK-NEXT:      }
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
    // CHECK-NEXT:        %[[VAL_239:.*]] = llvm.mlir.addressof @__builtin_var_NumWorkgroups__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_240:.*]] = llvm.load %[[VAL_239]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_241:.*]] = llvm.mlir.null : !llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_242:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_243:.*]] = llvm.getelementptr %[[VAL_241]][1] : (!llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>) -> !llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>
    // CHECK-NEXT:        %[[VAL_244:.*]] = llvm.ptrtoint %[[VAL_243]] : !llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>> to i64
    // CHECK-NEXT:        %[[VAL_245:.*]] = llvm.alloca %[[VAL_244]] x !llvm.struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)> : (i64) -> !llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>
    // CHECK-DAG:         %[[VAL_246:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_247:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:         %[[VAL_248:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_249:.*]] = llvm.extractelement %[[VAL_240]]{{\[}}%[[VAL_248]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_250:.*]] = llvm.getelementptr inbounds %[[VAL_245]][0, 0, 0, 0] : (!llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>) -> !llvm.ptr<i64, 4>
    // CHECK-NEXT:        llvm.store %[[VAL_249]], %[[VAL_250]] : !llvm.ptr<i64, 4>
    // CHECK-DAG:         %[[VAL_251:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:         %[[VAL_252:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:        %[[VAL_253:.*]] = llvm.extractelement %[[VAL_240]]{{\[}}%[[VAL_252]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_254:.*]] = llvm.getelementptr inbounds %[[VAL_245]][0, 0, 0, 1] : (!llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>) -> !llvm.ptr<i64, 4>
    // CHECK-NEXT:        llvm.store %[[VAL_253]], %[[VAL_254]] : !llvm.ptr<i64, 4>
    // CHECK-NEXT:        %[[VAL_255:.*]] = llvm.load %[[VAL_245]] : !llvm.ptr<struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>>
    // CHECK-NEXT:        llvm.return %[[VAL_255]] : !llvm.struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>
    // CHECK-NEXT:      }
    func.func @test_num_work_groups() -> !sycl_range_2_ {
      %0 = sycl.num_work_groups : !sycl_range_2_
      return %0 : !sycl_range_2_
    }

    // CHECK-LABEL:     llvm.func @test_num_work_groups_dim(
    // CHECK-SAME:                                          %[[VAL_256:.*]]: i32) -> i64 {
    // CHECK-NEXT:        %[[VAL_257:.*]] = llvm.mlir.addressof @__builtin_var_NumWorkgroups__ : !llvm.ptr<vector<3xi64>>
    // CHECK-NEXT:        %[[VAL_258:.*]] = llvm.load %[[VAL_257]] : !llvm.ptr<vector<3xi64>>
    // CHECK-DAG:         %[[VAL_259:.*]] = llvm.mlir.constant(dense<0> : vector<3xindex>) : vector<3xi64>
    // CHECK-DAG:         %[[VAL_260:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_261:.*]] = llvm.extractelement %[[VAL_258]]{{\[}}%[[VAL_260]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_262:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT:        %[[VAL_263:.*]] = llvm.insertelement %[[VAL_261]], %[[VAL_259]]{{\[}}%[[VAL_262]] : i64] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_264:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:        %[[VAL_265:.*]] = llvm.extractelement %[[VAL_258]]{{\[}}%[[VAL_264]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_266:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-NEXT:        %[[VAL_267:.*]] = llvm.insertelement %[[VAL_265]], %[[VAL_263]]{{\[}}%[[VAL_266]] : i64] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_268:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT:        %[[VAL_269:.*]] = llvm.extractelement %[[VAL_258]]{{\[}}%[[VAL_268]] : i32] : vector<3xi64>
    // CHECK-DAG:         %[[VAL_270:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK-NEXT:        %[[VAL_271:.*]] = llvm.insertelement %[[VAL_269]], %[[VAL_267]]{{\[}}%[[VAL_270]] : i64] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_272:.*]] = llvm.extractelement %[[VAL_271]]{{\[}}%[[VAL_256]] : i32] : vector<3xi64>
    // CHECK-NEXT:        llvm.return %[[VAL_272]] : i64
    // CHECK-NEXT:      }
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
