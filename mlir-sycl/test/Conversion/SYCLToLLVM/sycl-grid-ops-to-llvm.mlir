// RUN: sycl-mlir-opt -convert-sycl-to-llvm %s -o - | FileCheck %s

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

module attributes {gpu.container_module} {
  // CHECK:   gpu.module @kernels {
    //CHECK-DAG:      llvm.mlir.global external constant @__spirv_BuiltInSubgroupLocalInvocationId() {addr_space = 1 : i32} : i32
    //CHECK-DAG:      llvm.mlir.global external constant @__spirv_BuiltInSubgroupMaxSize() {addr_space = 1 : i32} : i32
    //CHECK-DAG:      llvm.mlir.global external constant @__spirv_BuiltInGlobalOffset() {addr_space = 1 : i32} : vector<3xi64>
    //CHECK-DAG:      llvm.mlir.global external constant @__spirv_BuiltInSubgroupId() {addr_space = 1 : i32} : i32
    //CHECK-DAG:      llvm.mlir.global external constant @__spirv_BuiltInSubgroupSize() {addr_space = 1 : i32} : i32
    //CHECK-DAG:      llvm.mlir.global external constant @__spirv_BuiltInNumSubgroups() {addr_space = 1 : i32} : i32
    //CHECK-DAG:      llvm.mlir.global external constant @__spirv_BuiltInWorkgroupId() {addr_space = 1 : i32} : vector<3xi64>
    //CHECK-DAG:      llvm.mlir.global external constant @__spirv_BuiltInWorkgroupSize() {addr_space = 1 : i32} : vector<3xi64>
    //CHECK-DAG:      llvm.mlir.global external constant @__spirv_BuiltInLocalInvocationId() {addr_space = 1 : i32} : vector<3xi64>
    //CHECK-DAG:      llvm.mlir.global external constant @__spirv_BuiltInGlobalInvocationId() {addr_space = 1 : i32} : vector<3xi64>
    //CHECK-DAG:      llvm.mlir.global external constant @__spirv_BuiltInNumWorkgroups() {addr_space = 1 : i32} : vector<3xi64>
    //CHECK-DAG:      llvm.mlir.global external constant @__spirv_BuiltInGlobalSize() {addr_space = 1 : i32} : vector<3xi64>
  gpu.module @kernels {
    // CHECK-LABEL:   llvm.func @test_num_work_items() -> !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> {
    // CHECK-NEXT:        %[[VAL_0:.*]] = llvm.mlir.addressof @__spirv_BuiltInGlobalSize : !llvm.ptr<1>
    // CHECK-NEXT:        %[[VAL_1:.*]] = llvm.load %[[VAL_0]] : !llvm.ptr<1> -> vector<3xi64>
    // CHECK-NEXT:        %[[VAL_2:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_3:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_6:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> : (i64) -> !llvm.ptr
    // CHECK-DAG:         %[[VAL_7:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_8:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:         %[[VAL_9:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_10:.*]] = llvm.extractelement %[[VAL_1]]{{\[}}%[[VAL_9]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_6]][0, 0, 0, %[[VAL_8]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
    // CHECK-NEXT:        %[[VAL_12:.*]] = llvm.getelementptr %[[VAL_11]]{{\[}}%[[VAL_7]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    // CHECK-NEXT:        llvm.store %[[VAL_10]], %[[VAL_12]] : i64, !llvm.ptr
    // CHECK-NEXT:        %[[VAL_13:.*]] = llvm.getelementptr %[[VAL_6]]{{\[}}%[[VAL_7]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
    // CHECK-NEXT:        %[[VAL_14:.*]] = llvm.load %[[VAL_13]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::range.1", {{.*}}>
    // CHECK-NEXT:        llvm.return %[[VAL_14]] : !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>
    // CHECK-NEXT:      }
    func.func @test_num_work_items() -> !sycl_range_1_ {
      %0 = sycl.num_work_items : !sycl_range_1_
      return %0 : !sycl_range_1_
    }

    // CHECK-LABEL:     llvm.func @test_global_id() -> !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)> {
    // CHECK-NEXT:        %[[VAL_32:.*]] = llvm.mlir.addressof @__spirv_BuiltInGlobalInvocationId : !llvm.ptr<1>
    // CHECK-NEXT:        %[[VAL_33:.*]] = llvm.load %[[VAL_32]] : !llvm.ptr<1> -> vector<3xi64>
    // CHECK-NEXT:        %[[VAL_34:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_35:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_38:.*]] = llvm.alloca %[[VAL_34]] x !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)> : (i64) -> !llvm.ptr
    // CHECK-DAG:         %[[VAL_39:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_40:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:         %[[VAL_41:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:        %[[VAL_42:.*]] = llvm.extractelement %[[VAL_33]]{{\[}}%[[VAL_41]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_43:.*]] = llvm.getelementptr inbounds %[[VAL_38]][0, 0, 0, %[[VAL_40]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.2", {{.*}}>
    // CHECK-NEXT:        %[[VAL_44:.*]] = llvm.getelementptr %[[VAL_43]]{{\[}}%[[VAL_39]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    // CHECK-NEXT:        llvm.store %[[VAL_42]], %[[VAL_44]] : i64, !llvm.ptr
    // CHECK-DAG:         %[[VAL_45:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:         %[[VAL_46:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_47:.*]] = llvm.extractelement %[[VAL_33]]{{\[}}%[[VAL_46]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_48:.*]] = llvm.getelementptr inbounds %[[VAL_38]][0, 0, 0, %[[VAL_45]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.2", {{.*}}>
    // CHECK-NEXT:        %[[VAL_49:.*]] = llvm.getelementptr %[[VAL_48]]{{\[}}%[[VAL_39]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    // CHECK-NEXT:        llvm.store %[[VAL_47]], %[[VAL_49]] : i64, !llvm.ptr
    // CHECK-NEXT:        %[[VAL_50:.*]] = llvm.getelementptr %[[VAL_38]]{{\[}}%[[VAL_39]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.2", {{.*}}>
    // CHECK-NEXT:        %[[VAL_51:.*]] = llvm.load %[[VAL_50]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::id.2", {{.*}}>
    // CHECK-NEXT:        llvm.return %[[VAL_51]] : !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>
    // CHECK-NEXT:      }
    func.func @test_global_id() -> !sycl_id_2_ {
      %0 = sycl.global_id : !sycl_id_2_
      return %0 : !sycl_id_2_
    }

    // CHECK-LABEL:     llvm.func @test_local_id() -> !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {
    // CHECK-NEXT:        %[[VAL_69:.*]] = llvm.mlir.addressof @__spirv_BuiltInLocalInvocationId : !llvm.ptr<1>
    // CHECK-NEXT:        %[[VAL_70:.*]] = llvm.load %[[VAL_69]] : !llvm.ptr<1> -> vector<3xi64>
    // CHECK-NEXT:        %[[VAL_71:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_72:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_75:.*]] = llvm.alloca %[[VAL_71]] x !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> : (i64) -> !llvm.ptr
    // CHECK-DAG:         %[[VAL_76:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_77:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:         %[[VAL_78:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT:        %[[VAL_79:.*]] = llvm.extractelement %[[VAL_70]]{{\[}}%[[VAL_78]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_80:.*]] = llvm.getelementptr inbounds %[[VAL_75]][0, 0, 0, %[[VAL_77]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
    // CHECK-NEXT:        %[[VAL_81:.*]] = llvm.getelementptr %[[VAL_80]]{{\[}}%[[VAL_76]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    // CHECK-NEXT:        llvm.store %[[VAL_79]], %[[VAL_81]] : i64, !llvm.ptr
    // CHECK-DAG:         %[[VAL_82:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:         %[[VAL_83:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:        %[[VAL_84:.*]] = llvm.extractelement %[[VAL_70]]{{\[}}%[[VAL_83]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_85:.*]] = llvm.getelementptr inbounds %[[VAL_75]][0, 0, 0, %[[VAL_82]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
    // CHECK-NEXT:        %[[VAL_86:.*]] = llvm.getelementptr %[[VAL_85]]{{\[}}%[[VAL_76]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    // CHECK-NEXT:        llvm.store %[[VAL_84]], %[[VAL_86]] : i64, !llvm.ptr
    // CHECK-DAG:         %[[VAL_87:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:         %[[VAL_88:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_89:.*]] = llvm.extractelement %[[VAL_70]]{{\[}}%[[VAL_88]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_90:.*]] = llvm.getelementptr inbounds %[[VAL_75]][0, 0, 0, %[[VAL_87]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
    // CHECK-NEXT:        %[[VAL_91:.*]] = llvm.getelementptr %[[VAL_90]]{{\[}}%[[VAL_76]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    // CHECK-NEXT:        llvm.store %[[VAL_89]], %[[VAL_91]] : i64, !llvm.ptr
    // CHECK-NEXT:        %[[VAL_92:.*]] = llvm.getelementptr %[[VAL_75]]{{\[}}%[[VAL_76]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
    // CHECK-NEXT:        %[[VAL_93:.*]] = llvm.load %[[VAL_92]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::id.3", {{.*}}>
    // CHECK-NEXT:        llvm.return %[[VAL_93]] : !llvm.struct<"class.sycl::_V1::id.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
    // CHECK-NEXT:      }
    func.func @test_local_id() -> !sycl_id_3_ {
      %0 = sycl.local_id : !sycl_id_3_
      return %0 : !sycl_id_3_
    }

    // CHECK-LABEL:     llvm.func @test_work_group_size() -> !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {
    // CHECK-NEXT:        %[[VAL_111:.*]] = llvm.mlir.addressof @__spirv_BuiltInWorkgroupSize : !llvm.ptr<1>
    // CHECK-NEXT:        %[[VAL_112:.*]] = llvm.load %[[VAL_111]] : !llvm.ptr<1> -> vector<3xi64>
    // CHECK-NEXT:        %[[VAL_113:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_114:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_117:.*]] = llvm.alloca %[[VAL_113]] x !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> : (i64) -> !llvm.ptr
    // CHECK-DAG:         %[[VAL_118:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_119:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:         %[[VAL_120:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT:        %[[VAL_121:.*]] = llvm.extractelement %[[VAL_112]]{{\[}}%[[VAL_120]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_122:.*]] = llvm.getelementptr inbounds %[[VAL_117]][0, 0, 0, %[[VAL_119]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.3", {{.*}}>
    // CHECK-NEXT:        %[[VAL_123:.*]] = llvm.getelementptr %[[VAL_122]]{{\[}}%[[VAL_118]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    // CHECK-NEXT:        llvm.store %[[VAL_121]], %[[VAL_123]] : i64, !llvm.ptr
    // CHECK-DAG:         %[[VAL_124:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:         %[[VAL_125:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:        %[[VAL_126:.*]] = llvm.extractelement %[[VAL_112]]{{\[}}%[[VAL_125]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_127:.*]] = llvm.getelementptr inbounds %[[VAL_117]][0, 0, 0, %[[VAL_124]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.3", {{.*}}>
    // CHECK-NEXT:        %[[VAL_128:.*]] = llvm.getelementptr %[[VAL_127]]{{\[}}%[[VAL_118]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    // CHECK-NEXT:        llvm.store %[[VAL_126]], %[[VAL_128]] : i64, !llvm.ptr
    // CHECK-DAG:         %[[VAL_129:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG:         %[[VAL_130:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_131:.*]] = llvm.extractelement %[[VAL_112]]{{\[}}%[[VAL_130]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_132:.*]] = llvm.getelementptr inbounds %[[VAL_117]][0, 0, 0, %[[VAL_129]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.3", {{.*}}>
    // CHECK-NEXT:        %[[VAL_133:.*]] = llvm.getelementptr %[[VAL_132]]{{\[}}%[[VAL_118]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    // CHECK-NEXT:        llvm.store %[[VAL_131]], %[[VAL_133]] : i64, !llvm.ptr
    // CHECK-NEXT:        %[[VAL_134:.*]] = llvm.getelementptr %[[VAL_117]]{{\[}}%[[VAL_118]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.3", {{.*}}>
    // CHECK-NEXT:        %[[VAL_135:.*]] = llvm.load %[[VAL_134]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::range.3", {{.*}}>
    // CHECK-NEXT:        llvm.return %[[VAL_135]] : !llvm.struct<"class.sycl::_V1::range.3", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)>
    // CHECK-NEXT:      }
    func.func @test_work_group_size() -> !sycl_range_3_ {
      %0 = sycl.work_group_size : !sycl_range_3_
      return %0 : !sycl_range_3_
    }

    // CHECK-LABEL:     llvm.func @test_work_group_id() -> !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> {
    // CHECK-NEXT:        %[[VAL_153:.*]] = llvm.mlir.addressof @__spirv_BuiltInWorkgroupId : !llvm.ptr<1>
    // CHECK-NEXT:        %[[VAL_154:.*]] = llvm.load %[[VAL_153]] : !llvm.ptr<1> -> vector<3xi64>
    // CHECK-NEXT:        %[[VAL_155:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_156:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_159:.*]] = llvm.alloca %[[VAL_155]] x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> : (i64) -> !llvm.ptr
    // CHECK-DAG:         %[[VAL_160:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_161:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:         %[[VAL_162:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_163:.*]] = llvm.extractelement %[[VAL_154]]{{\[}}%[[VAL_162]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_164:.*]] = llvm.getelementptr inbounds %[[VAL_159]][0, 0, 0, %[[VAL_161]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
    // CHECK-NEXT:        %[[VAL_165:.*]] = llvm.getelementptr %[[VAL_164]]{{\[}}%[[VAL_160]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    // CHECK-NEXT:        llvm.store %[[VAL_163]], %[[VAL_165]] : i64, !llvm.ptr
    // CHECK-NEXT:        %[[VAL_166:.*]] = llvm.getelementptr %[[VAL_159]]{{\[}}%[[VAL_160]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
    // CHECK-NEXT:        %[[VAL_167:.*]] = llvm.load %[[VAL_166]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
    // CHECK-NEXT:        llvm.return %[[VAL_167]] : !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>
    // CHECK-NEXT:      }
    func.func @test_work_group_id() -> !sycl_id_1_ {
      %0 = sycl.work_group_id : !sycl_id_1_
      return %0 : !sycl_id_1_
    }

    // CHECK-LABEL:     llvm.func @test_num_sub_groups() -> i32 {
    // CHECK-NEXT:        %[[VAL_200:.*]] = llvm.mlir.addressof @__spirv_BuiltInNumSubgroups : !llvm.ptr<1>
    // CHECK-NEXT:        %[[VAL_201:.*]] = llvm.load %[[VAL_200]] : !llvm.ptr<1> -> i32
    // CHECK-NEXT:        llvm.return %[[VAL_201]] : i32
    // CHECK-NEXT:      }
    func.func @test_num_sub_groups() -> i32 {
      %0 = sycl.num_sub_groups : i32
      return %0 : i32
    }

    // CHECK-LABEL:     llvm.func @test_sub_group_size() -> i32 {
    // CHECK-NEXT:        %[[VAL_203:.*]] = llvm.mlir.addressof @__spirv_BuiltInSubgroupSize : !llvm.ptr<1>
    // CHECK-NEXT:        %[[VAL_204:.*]] = llvm.load %[[VAL_203]] : !llvm.ptr<1> -> i32
    // CHECK-NEXT:        llvm.return %[[VAL_204]] : i32
    // CHECK-NEXT:      }
    func.func @test_sub_group_size() -> i32 {
      %0 = sycl.sub_group_size : i32
      return %0 : i32
    }

    // CHECK-LABEL:     llvm.func @test_sub_group_id() -> i32 {
    // CHECK-NEXT:        %[[VAL_206:.*]] = llvm.mlir.addressof @__spirv_BuiltInSubgroupId : !llvm.ptr<1>
    // CHECK-NEXT:        %[[VAL_207:.*]] = llvm.load %[[VAL_206]] : !llvm.ptr<1> -> i32
    // CHECK-NEXT:        llvm.return %[[VAL_207]] : i32
    // CHECK-NEXT:      }
    func.func @test_sub_group_id() -> i32 {
      %0 = sycl.sub_group_id : i32
      return %0 : i32
    }

    // CHECK-LABEL:     llvm.func @test_global_offset() -> !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> {
    // CHECK-NEXT:        %[[VAL_194:.*]] = llvm.mlir.addressof @__spirv_BuiltInGlobalOffset : !llvm.ptr<1>
    // CHECK-NEXT:        %[[VAL_195:.*]] = llvm.load %[[VAL_194]] : !llvm.ptr<1> -> vector<3xi64>
    // CHECK-NEXT:        %[[VAL_196:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_197:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_200:.*]] = llvm.alloca %[[VAL_196]] x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)> : (i64) -> !llvm.ptr
    // CHECK-DAG:         %[[VAL_201:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_202:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:         %[[VAL_203:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_204:.*]] = llvm.extractelement %[[VAL_195]]{{\[}}%[[VAL_203]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_205:.*]] = llvm.getelementptr inbounds %[[VAL_200]][0, 0, 0, %[[VAL_202]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
    // CHECK-NEXT:        %[[VAL_206:.*]] = llvm.getelementptr %[[VAL_205]]{{\[}}%[[VAL_201]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    // CHECK-NEXT:        llvm.store %[[VAL_204]], %[[VAL_206]] : i64, !llvm.ptr
    // CHECK-NEXT:        %[[VAL_207:.*]] = llvm.getelementptr %[[VAL_200]]{{\[}}%[[VAL_201]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
    // CHECK-NEXT:        %[[VAL_208:.*]] = llvm.load %[[VAL_207]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::id.1", {{.*}}>
    // CHECK-NEXT:        llvm.return %[[VAL_208]] : !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>
    // CHECK-NEXT:      }
    func.func @test_global_offset() -> !sycl_id_1_ {
      %0 = sycl.global_offset : !sycl_id_1_
      return %0 : !sycl_id_1_
    }

    // CHECK-LABEL:     llvm.func @test_num_work_groups() -> !llvm.struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)> {
    // CHECK-NEXT:        %[[VAL_226:.*]] = llvm.mlir.addressof @__spirv_BuiltInNumWorkgroups : !llvm.ptr<1>
    // CHECK-NEXT:        %[[VAL_227:.*]] = llvm.load %[[VAL_226]] : !llvm.ptr<1> -> vector<3xi64>
    // CHECK-NEXT:        %[[VAL_228:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_229:.*]] = llvm.mlir.constant(1 : index) : i64
    // CHECK-NEXT:        %[[VAL_232:.*]] = llvm.alloca %[[VAL_228]] x !llvm.struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)> : (i64) -> !llvm.ptr
    // CHECK-DAG:         %[[VAL_233:.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-DAG:         %[[VAL_234:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:         %[[VAL_235:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:        %[[VAL_236:.*]] = llvm.extractelement %[[VAL_227]]{{\[}}%[[VAL_235]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_237:.*]] = llvm.getelementptr inbounds %[[VAL_232]][0, 0, 0, %[[VAL_234]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.2", {{.*}}>
    // CHECK-NEXT:        %[[VAL_238:.*]] = llvm.getelementptr %[[VAL_237]]{{\[}}%[[VAL_233]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    // CHECK-NEXT:        llvm.store %[[VAL_236]], %[[VAL_238]] : i64, !llvm.ptr
    // CHECK-DAG:         %[[VAL_239:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG:         %[[VAL_240:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:        %[[VAL_241:.*]] = llvm.extractelement %[[VAL_227]]{{\[}}%[[VAL_240]] : i32] : vector<3xi64>
    // CHECK-NEXT:        %[[VAL_242:.*]] = llvm.getelementptr inbounds %[[VAL_232]][0, 0, 0, %[[VAL_239]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.2", {{.*}}>
    // CHECK-NEXT:        %[[VAL_243:.*]] = llvm.getelementptr %[[VAL_242]]{{\[}}%[[VAL_233]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    // CHECK-NEXT:        llvm.store %[[VAL_241]], %[[VAL_243]] : i64, !llvm.ptr
    // CHECK-NEXT:        %[[VAL_244:.*]] = llvm.getelementptr %[[VAL_232]]{{\[}}%[[VAL_233]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::range.2", {{.*}}>
    // CHECK-NEXT:        %[[VAL_245:.*]] = llvm.load %[[VAL_244]] : !llvm.ptr -> !llvm.struct<"class.sycl::_V1::range.2", {{.*}}>
    // CHECK-NEXT:        llvm.return %[[VAL_245]] : !llvm.struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<2 x i64>)>)>
    // CHECK-NEXT:      }
    func.func @test_num_work_groups() -> !sycl_range_2_ {
      %0 = sycl.num_work_groups : !sycl_range_2_
      return %0 : !sycl_range_2_
    }

    // CHECK-LABEL:     llvm.func @test_sub_group_max_size() -> i32 {
    // CHECK-NEXT:        %[[VAL_273:.*]] = llvm.mlir.addressof @__spirv_BuiltInSubgroupMaxSize : !llvm.ptr<1>
    // CHECK-NEXT:        %[[VAL_274:.*]] = llvm.load %[[VAL_273]] : !llvm.ptr<1> -> i32
    // CHECK-NEXT:        llvm.return %[[VAL_274]] : i32
    func.func @test_sub_group_max_size() -> i32 {
      %0 = sycl.sub_group_max_size : i32
      return %0 : i32
    }

    // CHECK-LABEL:     llvm.func @test_sub_group_local_id() -> i32 {
    // CHECK-NEXT:        %[[VAL_276:.*]] = llvm.mlir.addressof @__spirv_BuiltInSubgroupLocalInvocationId : !llvm.ptr<1>
    // CHECK-NEXT:        %[[VAL_277:.*]] = llvm.load %[[VAL_276]] : !llvm.ptr<1> -> i32
    // CHECK-NEXT:        llvm.return %[[VAL_277]] : i32
    // CHECK-NEXT:      }
    func.func @test_sub_group_local_id() -> i32 {
      %0 = sycl.sub_group_local_id : i32
      return %0 : i32
    }
  }
}
