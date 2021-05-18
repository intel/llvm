; ModuleID = 'matrix-bf16-test-sycl-spir64-unknown-unknown-sycldevice.bc'
source_filename = "matrix-bf16-test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown-sycldevice"

%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" = type { [2 x i64] }
%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" }
%class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon = type { %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", i64, i64, i64 }
%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" = type { %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice", %union._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEUt_E.anon }
%"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" = type { %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id", %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" }
%union._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEUt_E.anon = type { i16 addrspace(1)* }
%"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" = type { %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice", %union._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEUt_E.anon }
%union._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEUt_E.anon = type { float addrspace(1)* }
%"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" = type { %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item", %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item", %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" }
%"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" = type { %"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb1EEE.cl::sycl::detail::ItemBase" }
%"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb1EEE.cl::sycl::detail::ItemBase" = type { %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" }
%"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" = type { %"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb0EEE.cl::sycl::detail::ItemBase" }
%"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb0EEE.cl::sycl::detail::ItemBase" = type { %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" }
%"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" = type { %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" }
%"class._ZTSN2cl4sycl6detail15accessor_commonItLi2ELNS0_6access4modeE1026ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::detail::accessor_common" = type { i8 }
%"class._ZTSN2cl4sycl6detail15accessor_commonIfLi2ELNS0_6access4modeE1026ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::detail::accessor_common" = type { i8 }
%"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" = type { %"struct._ZTSN5__spv19__spirv_MatrixINTELIfLm47ELm47ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* }
%"struct._ZTSN5__spv19__spirv_MatrixINTELIfLm47ELm47ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" = type opaque
%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" = type { float addrspace(1)* }
%"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" = type { i8 }
%"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" = type { %"struct._ZTSN5__spv19__spirv_MatrixINTELItLm47ELm146ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* }
%"struct._ZTSN5__spv19__spirv_MatrixINTELItLm47ELm146ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" = type opaque
%"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" = type { %"struct._ZTSN5__spv19__spirv_MatrixINTELItLm146ELm47ELNS_12MatrixLayoutE3ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* }
%"struct._ZTSN5__spv19__spirv_MatrixINTELItLm146ELm47ELNS_12MatrixLayoutE3ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" = type opaque
%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" = type { i16 addrspace(1)* }

$_ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_E7imatrix = comdat any

$_ZN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEC2Ev = comdat any

$_ZN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEC2Ev = comdat any

$_ZN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE6__initEPU3AS1tNS0_5rangeILi2EEESD_NS0_2idILi2EEE = comdat any

$_ZN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE6__initEPU3AS1fNS0_5rangeILi2EEESD_NS0_2idILi2EEE = comdat any

$_ZZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_ENKUlNSA_7nd_itemILi2EEEE_clESF_ = comdat any

$_ZN2cl4sycl6detail7Builder10getElementILi2EEEKNS0_7nd_itemIXT_EEEPS5_ = comdat any

$_ZN2cl4sycl6detail7declptrINS0_7nd_itemILi2EEEEEPT_v = comdat any

$_ZN2cl4sycl2idILi2EEC2Ev = comdat any

$_ZN2cl4sycl6detail14InitializedValILi2ENS0_5rangeEE3getILi0EEENS3_ILi2EEEv = comdat any

$_ZN2cl4sycl6detail18AccessorImplDeviceILi2EEC2ENS0_2idILi2EEENS0_5rangeILi2EEES7_ = comdat any

$_ZN2cl4sycl6detail5arrayILi2EEC2ILi2ELm0EEEv = comdat any

$_ZN2cl4sycl6detail5arrayILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm = comdat any

$_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm = comdat any

$_ZN2cl4sycl6detail5arrayILi2EEixEi = comdat any

$_ZN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE9getOffsetEv = comdat any

$_ZN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getAccessRangeEv = comdat any

$_ZN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getMemoryRangeEv = comdat any

$_ZN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE9getOffsetEv = comdat any

$_ZN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getAccessRangeEv = comdat any

$_ZN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getMemoryRangeEv = comdat any

$_ZNK2cl4sycl7nd_itemILi2EE13get_sub_groupEv = comdat any

$_ZN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEC2ES7_ = comdat any

$_ZN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEC2ES7_ = comdat any

$_ZN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEC2ES7_ = comdat any

$_ZNK2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE11get_pointerILS4_2014EvEENS0_9multi_ptrIfLNS2_13address_spaceE1EEEv = comdat any

$_ZNK2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEplEl = comdat any

$_ZNK2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE11get_pointerILS4_2014EvEENS0_9multi_ptrItLNS2_13address_spaceE1EEEv = comdat any

$_ZNK2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEplEl = comdat any

$_ZNK2cl4sycl6detail5arrayILi2EEixEi = comdat any

$_ZNK2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EE3getEv = comdat any

$_ZNK2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getLinearIndexILi2EEEmNS0_2idIXT_EEE = comdat any

$_ZNK2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE15getQualifiedPtrEv = comdat any

$_ZN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEC2EPU3AS1f = comdat any

$_ZNK2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getMemoryRangeEv = comdat any

$_ZNK2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE9getOffsetEv = comdat any

$_ZNK2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EE3getEv = comdat any

$_ZNK2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getLinearIndexILi2EEEmNS0_2idIXT_EEE = comdat any

$_ZNK2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE15getQualifiedPtrEv = comdat any

$_ZN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEC2EPU3AS1t = comdat any

$_ZNK2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getMemoryRangeEv = comdat any

$_ZNK2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE9getOffsetEv = comdat any

$_ZN2cl4sycl6detail7Builder11createGroupILi2EEENS0_5groupIXT_EEERKNS0_5rangeIXT_EEES9_S9_RKNS0_2idIXT_EEE = comdat any

$_ZN2cl4sycl6detail7Builder10createItemILi2ELb1EEENSt9enable_ifIXT0_ENS0_4itemIXT_EXT0_EEEE4typeERKNS0_5rangeIXT_EEERKNS0_2idIXT_EEESG_ = comdat any

$_ZN2cl4sycl6detail7Builder10createItemILi2ELb0EEENSt9enable_ifIXntT0_ENS0_4itemIXT_EXT0_EEEE4typeERKNS0_5rangeIXT_EEERKNS0_2idIXT_EEE = comdat any

$_ZN2cl4sycl6detail7Builder12createNDItemILi2EEENS0_7nd_itemIXT_EEERKNS0_4itemIXT_ELb1EEERKNS6_IXT_ELb0EEERKNS0_5groupIXT_EEE = comdat any

$_ZN7__spirv21InitSizesSTGlobalSizeILi2EN2cl4sycl5rangeILi2EEEE8initSizeEv = comdat any

$_Z20__spirv_GlobalSize_yv = comdat any

$_Z20__spirv_GlobalSize_xv = comdat any

$_ZN7__spirv24InitSizesSTWorkgroupSizeILi2EN2cl4sycl5rangeILi2EEEE8initSizeEv = comdat any

$_Z23__spirv_WorkgroupSize_yv = comdat any

$_Z23__spirv_WorkgroupSize_xv = comdat any

$_ZN7__spirv24InitSizesSTNumWorkgroupsILi2EN2cl4sycl5rangeILi2EEEE8initSizeEv = comdat any

$_Z23__spirv_NumWorkgroups_yv = comdat any

$_Z23__spirv_NumWorkgroups_xv = comdat any

$_ZN7__spirv22InitSizesSTWorkgroupIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv = comdat any

$_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm = comdat any

$_Z21__spirv_WorkgroupId_yv = comdat any

$_Z21__spirv_WorkgroupId_xv = comdat any

$_ZN7__spirv29InitSizesSTGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv = comdat any

$_Z28__spirv_GlobalInvocationId_yv = comdat any

$_Z28__spirv_GlobalInvocationId_xv = comdat any

$_ZN7__spirv28InitSizesSTLocalInvocationIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv = comdat any

$_Z27__spirv_LocalInvocationId_yv = comdat any

$_Z27__spirv_LocalInvocationId_xv = comdat any

$_ZN7__spirv23InitSizesSTGlobalOffsetILi2EN2cl4sycl2idILi2EEEE8initSizeEv = comdat any

$_Z22__spirv_GlobalOffset_yv = comdat any

$_Z22__spirv_GlobalOffset_xv = comdat any

$_ZN2cl4sycl5groupILi2EEC2ERKNS0_5rangeILi2EEES6_S4_RKNS0_2idILi2EEE = comdat any

$_ZN2cl4sycl4itemILi2ELb1EEC2ILb1EEERNSt9enable_ifIXT_EKNS0_5rangeILi2EEEE4typeERKNS0_2idILi2EEESE_ = comdat any

$_ZN2cl4sycl4itemILi2ELb0EEC2ILb0EEERNSt9enable_ifIXntT_EKNS0_5rangeILi2EEEE4typeERKNS0_2idILi2EEE = comdat any

$_ZN2cl4sycl7nd_itemILi2EEC2ERKNS0_4itemILi2ELb1EEERKNS3_ILi2ELb0EEERKNS0_5groupILi2EEE = comdat any

@__spirv_BuiltInGlobalSize = external dso_local addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInWorkgroupSize = external dso_local addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInNumWorkgroups = external dso_local addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInWorkgroupId = external dso_local addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInGlobalInvocationId = external dso_local addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInLocalInvocationId = external dso_local addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInGlobalOffset = external dso_local addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent norecurse mustprogress
define weak_odr dso_local spir_kernel void @_ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_E7imatrix(i16 addrspace(1)* %_arg_, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %_arg_1, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %_arg_2, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %_arg_3, i16 addrspace(1)* %_arg_4, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %_arg_6, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %_arg_7, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %_arg_8, float addrspace(1)* %_arg_9, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %_arg_11, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %_arg_12, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %_arg_13, i64 %_arg_14, i64 %_arg_16, i64 %_arg_18) #0 comdat !kernel_arg_buffer_location !6 {
entry:
  %_arg_.addr = alloca i16 addrspace(1)*, align 8
  %_arg_.addr.ascast = addrspacecast i16 addrspace(1)** %_arg_.addr to i16 addrspace(1)* addrspace(4)*
  %_arg_.addr5 = alloca i16 addrspace(1)*, align 8
  %_arg_.addr5.ascast = addrspacecast i16 addrspace(1)** %_arg_.addr5 to i16 addrspace(1)* addrspace(4)*
  %_arg_.addr10 = alloca float addrspace(1)*, align 8
  %_arg_.addr10.ascast = addrspacecast float addrspace(1)** %_arg_.addr10 to float addrspace(1)* addrspace(4)*
  %_arg_.addr15 = alloca i64, align 8
  %_arg_.addr15.ascast = addrspacecast i64* %_arg_.addr15 to i64 addrspace(4)*
  %_arg_.addr17 = alloca i64, align 8
  %_arg_.addr17.ascast = addrspacecast i64* %_arg_.addr17 to i64 addrspace(4)*
  %_arg_.addr19 = alloca i64, align 8
  %_arg_.addr19.ascast = addrspacecast i64* %_arg_.addr19 to i64 addrspace(4)*
  %0 = alloca %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, align 8
  %1 = addrspacecast %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon* %0 to %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)*
  %agg.tmp = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", align 8
  %agg.tmp.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %agg.tmp to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %agg.tmp20 = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", align 8
  %agg.tmp20.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %agg.tmp20 to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %agg.tmp21 = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id", align 8
  %agg.tmp21.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %agg.tmp21 to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*
  %agg.tmp22 = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", align 8
  %agg.tmp22.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %agg.tmp22 to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %agg.tmp23 = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", align 8
  %agg.tmp23.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %agg.tmp23 to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %agg.tmp24 = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id", align 8
  %agg.tmp24.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %agg.tmp24 to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*
  %agg.tmp25 = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", align 8
  %agg.tmp25.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %agg.tmp25 to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %agg.tmp26 = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", align 8
  %agg.tmp26.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %agg.tmp26 to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %agg.tmp27 = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id", align 8
  %agg.tmp27.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %agg.tmp27 to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*
  %agg.tmp28 = alloca %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item", align 8
  %agg.tmp28.ascast = addrspacecast %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item"* %agg.tmp28 to %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)*
  store i16 addrspace(1)* %_arg_, i16 addrspace(1)* addrspace(4)* %_arg_.addr.ascast, align 8, !tbaa !7
  %_arg_1.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %_arg_1 to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %_arg_2.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %_arg_2 to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %_arg_3.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %_arg_3 to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*
  store i16 addrspace(1)* %_arg_4, i16 addrspace(1)* addrspace(4)* %_arg_.addr5.ascast, align 8, !tbaa !7
  %_arg_6.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %_arg_6 to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %_arg_7.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %_arg_7 to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %_arg_8.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %_arg_8 to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*
  store float addrspace(1)* %_arg_9, float addrspace(1)* addrspace(4)* %_arg_.addr10.ascast, align 8, !tbaa !7
  %_arg_11.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %_arg_11 to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %_arg_12.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %_arg_12 to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %_arg_13.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %_arg_13 to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*
  store i64 %_arg_14, i64 addrspace(4)* %_arg_.addr15.ascast, align 8, !tbaa !11
  store i64 %_arg_16, i64 addrspace(4)* %_arg_.addr17.ascast, align 8, !tbaa !11
  store i64 %_arg_18, i64 addrspace(4)* %_arg_.addr19.ascast, align 8, !tbaa !11
  %2 = bitcast %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon* %0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 192, i8* %2) #12
  %3 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %1, i32 0, i32 0
  call spir_func void @_ZN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEC2Ev(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %3) #13
  %4 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %1, i32 0, i32 1
  call spir_func void @_ZN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEC2Ev(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %4) #13
  %5 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %1, i32 0, i32 2
  call spir_func void @_ZN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEC2Ev(%"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %5) #13
  %6 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %1, i32 0, i32 3
  %7 = load i64, i64 addrspace(4)* %_arg_.addr15.ascast, align 8, !tbaa !11
  store i64 %7, i64 addrspace(4)* %6, align 8, !tbaa !13
  %8 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %1, i32 0, i32 4
  %9 = load i64, i64 addrspace(4)* %_arg_.addr17.ascast, align 8, !tbaa !11
  store i64 %9, i64 addrspace(4)* %8, align 8, !tbaa !20
  %10 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %1, i32 0, i32 5
  %11 = load i64, i64 addrspace(4)* %_arg_.addr19.ascast, align 8, !tbaa !11
  store i64 %11, i64 addrspace(4)* %10, align 8, !tbaa !21
  %12 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %1, i32 0, i32 0
  %13 = load i16 addrspace(1)*, i16 addrspace(1)* addrspace(4)* %_arg_.addr.ascast, align 8, !tbaa !7
  %14 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %agg.tmp.ascast to i8 addrspace(4)*
  %15 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %_arg_1.ascast to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %14, i8 addrspace(4)* align 8 %15, i64 16, i1 false)
  %16 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %agg.tmp20.ascast to i8 addrspace(4)*
  %17 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %_arg_2.ascast to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %16, i8 addrspace(4)* align 8 %17, i64 16, i1 false)
  %18 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %agg.tmp21.ascast to i8 addrspace(4)*
  %19 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %_arg_3.ascast to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %18, i8 addrspace(4)* align 8 %19, i64 16, i1 false)
  %agg.tmp.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %agg.tmp.ascast to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"*
  %agg.tmp20.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %agg.tmp20.ascast to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"*
  %agg.tmp21.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %agg.tmp21.ascast to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"*
  call spir_func void @_ZN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE6__initEPU3AS1tNS0_5rangeILi2EEESD_NS0_2idILi2EEE(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %12, i16 addrspace(1)* %13, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.tmp.ascast.ascast, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.tmp20.ascast.ascast, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %agg.tmp21.ascast.ascast) #13
  %20 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %1, i32 0, i32 1
  %21 = load i16 addrspace(1)*, i16 addrspace(1)* addrspace(4)* %_arg_.addr5.ascast, align 8, !tbaa !7
  %22 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %agg.tmp22.ascast to i8 addrspace(4)*
  %23 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %_arg_6.ascast to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %22, i8 addrspace(4)* align 8 %23, i64 16, i1 false)
  %24 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %agg.tmp23.ascast to i8 addrspace(4)*
  %25 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %_arg_7.ascast to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %24, i8 addrspace(4)* align 8 %25, i64 16, i1 false)
  %26 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %agg.tmp24.ascast to i8 addrspace(4)*
  %27 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %_arg_8.ascast to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %26, i8 addrspace(4)* align 8 %27, i64 16, i1 false)
  %agg.tmp22.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %agg.tmp22.ascast to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"*
  %agg.tmp23.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %agg.tmp23.ascast to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"*
  %agg.tmp24.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %agg.tmp24.ascast to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"*
  call spir_func void @_ZN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE6__initEPU3AS1tNS0_5rangeILi2EEESD_NS0_2idILi2EEE(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %20, i16 addrspace(1)* %21, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.tmp22.ascast.ascast, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.tmp23.ascast.ascast, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %agg.tmp24.ascast.ascast) #13
  %28 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %1, i32 0, i32 2
  %29 = load float addrspace(1)*, float addrspace(1)* addrspace(4)* %_arg_.addr10.ascast, align 8, !tbaa !7
  %30 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %agg.tmp25.ascast to i8 addrspace(4)*
  %31 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %_arg_11.ascast to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %30, i8 addrspace(4)* align 8 %31, i64 16, i1 false)
  %32 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %agg.tmp26.ascast to i8 addrspace(4)*
  %33 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %_arg_12.ascast to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %32, i8 addrspace(4)* align 8 %33, i64 16, i1 false)
  %34 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %agg.tmp27.ascast to i8 addrspace(4)*
  %35 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %_arg_13.ascast to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %34, i8 addrspace(4)* align 8 %35, i64 16, i1 false)
  %agg.tmp25.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %agg.tmp25.ascast to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"*
  %agg.tmp26.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %agg.tmp26.ascast to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"*
  %agg.tmp27.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %agg.tmp27.ascast to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"*
  call spir_func void @_ZN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE6__initEPU3AS1fNS0_5rangeILi2EEESD_NS0_2idILi2EEE(%"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %28, float addrspace(1)* %29, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.tmp25.ascast.ascast, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.tmp26.ascast.ascast, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %agg.tmp27.ascast.ascast) #13
  %call = call spir_func %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* @_ZN2cl4sycl6detail7declptrINS0_7nd_itemILi2EEEEEPT_v() #13
  call spir_func void @_ZN2cl4sycl6detail7Builder10getElementILi2EEEKNS0_7nd_itemIXT_EEEPS5_(%"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* sret(%"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item") align 8 %agg.tmp28.ascast, %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* %call) #13
  %agg.tmp28.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* %agg.tmp28.ascast to %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item"*
  call spir_func void @_ZZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_ENKUlNSA_7nd_itemILi2EEEE_clESF_(%class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* align 8 dereferenceable_or_null(192) %1, %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item"* byval(%"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item") align 8 %agg.tmp28.ascast.ascast) #13
  %36 = bitcast %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon* %0 to i8*
  call void @llvm.lifetime.end.p0i8(i64 192, i8* %36) #12
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: convergent norecurse
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEC2Ev(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this) unnamed_addr #2 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  %agg.tmp = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id", align 8
  %agg.tmp.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %agg.tmp to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*
  %agg.tmp2 = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", align 8
  %agg.tmp2.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %agg.tmp2 to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %agg.tmp3 = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", align 8
  %agg.tmp3.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %agg.tmp3 to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  store %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = bitcast %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1 to %"class._ZTSN2cl4sycl6detail15accessor_commonItLi2ELNS0_6access4modeE1026ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::detail::accessor_common" addrspace(4)*
  %impl = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %1 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %agg.tmp.ascast to i8 addrspace(4)*
  call void @llvm.memset.p4i8.i64(i8 addrspace(4)* align 8 %1, i8 0, i64 16, i1 false)
  call spir_func void @_ZN2cl4sycl2idILi2EEC2Ev(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable_or_null(16) %agg.tmp.ascast) #13
  call spir_func void @_ZN2cl4sycl6detail14InitializedValILi2ENS0_5rangeEE3getILi0EEENS3_ILi2EEEv(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* sret(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.tmp2.ascast) #13
  call spir_func void @_ZN2cl4sycl6detail14InitializedValILi2ENS0_5rangeEE3getILi0EEENS3_ILi2EEEv(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* sret(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.tmp3.ascast) #13
  %agg.tmp.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %agg.tmp.ascast to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"*
  %agg.tmp2.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %agg.tmp2.ascast to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"*
  %agg.tmp3.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %agg.tmp3.ascast to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"*
  call spir_func void @_ZN2cl4sycl6detail18AccessorImplDeviceILi2EEC2ENS0_2idILi2EEENS0_5rangeILi2EEES7_(%"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* align 8 dereferenceable_or_null(48) %impl, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %agg.tmp.ascast.ascast, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.tmp2.ascast.ascast, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.tmp3.ascast.ascast) #13
  ret void
}

; Function Attrs: convergent norecurse
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEC2Ev(%"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this) unnamed_addr #2 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  %agg.tmp = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id", align 8
  %agg.tmp.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %agg.tmp to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*
  %agg.tmp2 = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", align 8
  %agg.tmp2.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %agg.tmp2 to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %agg.tmp3 = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", align 8
  %agg.tmp3.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %agg.tmp3 to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  store %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = bitcast %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1 to %"class._ZTSN2cl4sycl6detail15accessor_commonIfLi2ELNS0_6access4modeE1026ELNS3_6targetE2014ELNS3_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::detail::accessor_common" addrspace(4)*
  %impl = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %1 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %agg.tmp.ascast to i8 addrspace(4)*
  call void @llvm.memset.p4i8.i64(i8 addrspace(4)* align 8 %1, i8 0, i64 16, i1 false)
  call spir_func void @_ZN2cl4sycl2idILi2EEC2Ev(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable_or_null(16) %agg.tmp.ascast) #13
  call spir_func void @_ZN2cl4sycl6detail14InitializedValILi2ENS0_5rangeEE3getILi0EEENS3_ILi2EEEv(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* sret(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.tmp2.ascast) #13
  call spir_func void @_ZN2cl4sycl6detail14InitializedValILi2ENS0_5rangeEE3getILi0EEENS3_ILi2EEEv(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* sret(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.tmp3.ascast) #13
  %agg.tmp.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %agg.tmp.ascast to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"*
  %agg.tmp2.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %agg.tmp2.ascast to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"*
  %agg.tmp3.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %agg.tmp3.ascast to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"*
  call spir_func void @_ZN2cl4sycl6detail18AccessorImplDeviceILi2EEC2ENS0_2idILi2EEENS0_5rangeILi2EEES7_(%"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* align 8 dereferenceable_or_null(48) %impl, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %agg.tmp.ascast.ascast, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.tmp2.ascast.ascast, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.tmp3.ascast.ascast) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE6__initEPU3AS1tNS0_5rangeILi2EEESD_NS0_2idILi2EEE(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this, i16 addrspace(1)* %Ptr, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %AccessRange, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %MemRange, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %Offset) #3 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  %Ptr.addr = alloca i16 addrspace(1)*, align 8
  %Ptr.addr.ascast = addrspacecast i16 addrspace(1)** %Ptr.addr to i16 addrspace(1)* addrspace(4)*
  %I = alloca i32, align 4
  %I.ascast = addrspacecast i32* %I to i32 addrspace(4)*
  %cleanup.dest.slot = alloca i32, align 4
  store %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  store i16 addrspace(1)* %Ptr, i16 addrspace(1)* addrspace(4)* %Ptr.addr.ascast, align 8, !tbaa !7
  %AccessRange.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %AccessRange to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %MemRange.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %MemRange to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %Offset.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %Offset to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*
  %this1 = load %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = load i16 addrspace(1)*, i16 addrspace(1)* addrspace(4)* %Ptr.addr.ascast, align 8, !tbaa !7
  %1 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 1
  %MData = bitcast %union._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEUt_E.anon addrspace(4)* %1 to i16 addrspace(1)* addrspace(4)*
  store i16 addrspace(1)* %0, i16 addrspace(1)* addrspace(4)* %MData, align 8, !tbaa !22
  %2 = bitcast i32* %I to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %2) #12
  store i32 0, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %3 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %cmp = icmp slt i32 %3, 2
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  %4 = bitcast i32* %I to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %4) #12
  br label %for.end

for.body:                                         ; preds = %for.cond
  %5 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %Offset.ascast to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %6 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %call = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %5, i32 %6) #13
  %7 = load i64, i64 addrspace(4)* %call, align 8, !tbaa !11
  %call2 = call spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* @_ZN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE9getOffsetEv(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this1) #13
  %8 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %call2 to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %9 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %call3 = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %8, i32 %9) #13
  store i64 %7, i64 addrspace(4)* %call3, align 8, !tbaa !11
  %10 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %AccessRange.ascast to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %11 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %call4 = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %10, i32 %11) #13
  %12 = load i64, i64 addrspace(4)* %call4, align 8, !tbaa !11
  %call5 = call spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* @_ZN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getAccessRangeEv(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this1) #13
  %13 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %call5 to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %14 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %call6 = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %13, i32 %14) #13
  store i64 %12, i64 addrspace(4)* %call6, align 8, !tbaa !11
  %15 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %MemRange.ascast to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %16 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %call7 = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %15, i32 %16) #13
  %17 = load i64, i64 addrspace(4)* %call7, align 8, !tbaa !11
  %call8 = call spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* @_ZN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getMemoryRangeEv(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this1) #13
  %18 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %call8 to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %19 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %call9 = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %18, i32 %19) #13
  store i64 %17, i64 addrspace(4)* %call9, align 8, !tbaa !11
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %20 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %inc = add nsw i32 %20, 1
  store i32 %inc, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  br label %for.cond, !llvm.loop !25

for.end:                                          ; preds = %for.cond.cleanup
  ret void
}

; Function Attrs: argmemonly nofree nounwind willreturn
declare void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* noalias nocapture writeonly, i8 addrspace(4)* noalias nocapture readonly, i64, i1 immarg) #4

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE6__initEPU3AS1fNS0_5rangeILi2EEESD_NS0_2idILi2EEE(%"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this, float addrspace(1)* %Ptr, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %AccessRange, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %MemRange, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %Offset) #3 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  %Ptr.addr = alloca float addrspace(1)*, align 8
  %Ptr.addr.ascast = addrspacecast float addrspace(1)** %Ptr.addr to float addrspace(1)* addrspace(4)*
  %I = alloca i32, align 4
  %I.ascast = addrspacecast i32* %I to i32 addrspace(4)*
  %cleanup.dest.slot = alloca i32, align 4
  store %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  store float addrspace(1)* %Ptr, float addrspace(1)* addrspace(4)* %Ptr.addr.ascast, align 8, !tbaa !7
  %AccessRange.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %AccessRange to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %MemRange.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %MemRange to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %Offset.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %Offset to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*
  %this1 = load %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = load float addrspace(1)*, float addrspace(1)* addrspace(4)* %Ptr.addr.ascast, align 8, !tbaa !7
  %1 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 1
  %MData = bitcast %union._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEUt_E.anon addrspace(4)* %1 to float addrspace(1)* addrspace(4)*
  store float addrspace(1)* %0, float addrspace(1)* addrspace(4)* %MData, align 8, !tbaa !22
  %2 = bitcast i32* %I to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %2) #12
  store i32 0, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %3 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %cmp = icmp slt i32 %3, 2
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  %4 = bitcast i32* %I to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %4) #12
  br label %for.end

for.body:                                         ; preds = %for.cond
  %5 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %Offset.ascast to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %6 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %call = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %5, i32 %6) #13
  %7 = load i64, i64 addrspace(4)* %call, align 8, !tbaa !11
  %call2 = call spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* @_ZN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE9getOffsetEv(%"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this1) #13
  %8 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %call2 to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %9 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %call3 = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %8, i32 %9) #13
  store i64 %7, i64 addrspace(4)* %call3, align 8, !tbaa !11
  %10 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %AccessRange.ascast to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %11 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %call4 = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %10, i32 %11) #13
  %12 = load i64, i64 addrspace(4)* %call4, align 8, !tbaa !11
  %call5 = call spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* @_ZN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getAccessRangeEv(%"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this1) #13
  %13 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %call5 to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %14 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %call6 = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %13, i32 %14) #13
  store i64 %12, i64 addrspace(4)* %call6, align 8, !tbaa !11
  %15 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %MemRange.ascast to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %16 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %call7 = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %15, i32 %16) #13
  %17 = load i64, i64 addrspace(4)* %call7, align 8, !tbaa !11
  %call8 = call spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* @_ZN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getMemoryRangeEv(%"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this1) #13
  %18 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %call8 to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %19 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %call9 = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %18, i32 %19) #13
  store i64 %17, i64 addrspace(4)* %call9, align 8, !tbaa !11
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %20 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %inc = add nsw i32 %20, 1
  store i32 %inc, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  br label %for.cond, !llvm.loop !28

for.end:                                          ; preds = %for.cond.cleanup
  ret void
}

; Function Attrs: convergent inlinehint norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_ENKUlNSA_7nd_itemILi2EEEE_clESF_(%class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* align 8 dereferenceable_or_null(192) %this, %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item"* byval(%"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item") align 8 %spmd_item) #5 comdat align 2 {
entry:
  %res.addr.i134 = alloca %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, align 8
  %stride.addr.i136 = alloca i64, align 8
  %L.addr.i138 = alloca i32, align 4
  %Ptr.i140 = alloca float addrspace(4)*, align 8
  %agg.tmp34.ascast.ascast133 = alloca %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", align 8
  %agg.tmp33.ascast.ascast132 = alloca %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group", align 8
  %mA.addr.i = alloca %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, align 8
  %mB.addr.i = alloca %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, align 8
  %mC.addr.i = alloca %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, align 8
  %agg.tmp.i = alloca %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group", align 1
  %agg.tmp32.ascast.ascast128 = alloca %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group", align 8
  %res.addr.i114 = alloca %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, align 8
  %stride.addr.i116 = alloca i64, align 8
  %L.addr.i118 = alloca i32, align 4
  %Ptr.i120 = alloca i16 addrspace(4)*, align 8
  %agg.tmp20.ascast.ascast113 = alloca %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", align 8
  %agg.tmp19.ascast.ascast112 = alloca %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group", align 8
  %res.addr.i98 = alloca %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, align 8
  %stride.addr.i100 = alloca i64, align 8
  %L.addr.i102 = alloca i32, align 4
  %Ptr.i104 = alloca i16 addrspace(4)*, align 8
  %agg.tmp12.ascast.ascast97 = alloca %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", align 8
  %agg.tmp11.ascast.ascast96 = alloca %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group", align 8
  %res.addr.i = alloca %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, align 8
  %stride.addr.i = alloca i64, align 8
  %L.addr.i = alloca i32, align 4
  %Ptr.i = alloca float addrspace(4)*, align 8
  %agg.tmp7.ascast.ascast95 = alloca %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", align 8
  %agg.tmp6.ascast.ascast94 = alloca %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group", align 8
  %retval.i.i71 = alloca i64, align 8
  %this.addr.i.i72 = alloca %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)*, align 8
  %dimension.addr.i.i73 = alloca i32, align 4
  %Id.i.i74 = alloca i64, align 8
  %retval.i75 = alloca i64, align 8
  %this.addr.i77 = alloca %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)*, align 8
  %dimension.addr.i79 = alloca i32, align 4
  %Id.i81 = alloca i64, align 8
  %retval.i.i47 = alloca i64, align 8
  %this.addr.i.i48 = alloca %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)*, align 8
  %dimension.addr.i.i49 = alloca i32, align 4
  %Id.i.i50 = alloca i64, align 8
  %retval.i51 = alloca i64, align 8
  %this.addr.i53 = alloca %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)*, align 8
  %dimension.addr.i55 = alloca i32, align 4
  %Id.i57 = alloca i64, align 8
  %retval.i.i = alloca i64, align 8
  %this.addr.i.i = alloca %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)*, align 8
  %dimension.addr.i.i = alloca i32, align 4
  %Id.i.i = alloca i64, align 8
  %retval.i = alloca i64, align 8
  %this.addr.i = alloca %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)*, align 8
  %dimension.addr.i = alloca i32, align 4
  %Id.i = alloca i64, align 8
  %this.addr = alloca %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)** %this.addr to %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* addrspace(4)*
  %global_idx = alloca i64, align 8
  %global_idx.ascast = addrspacecast i64* %global_idx to i64 addrspace(4)*
  %global_idy = alloca i64, align 8
  %global_idy.ascast = addrspacecast i64* %global_idy to i64 addrspace(4)*
  %sg_startx = alloca i64, align 8
  %sg_startx.ascast = addrspacecast i64* %sg_startx to i64 addrspace(4)*
  %sg_starty = alloca i64, align 8
  %sg_starty.ascast = addrspacecast i64* %sg_starty to i64 addrspace(4)*
  %sg = alloca %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group", align 1
  %sg.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %sg to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)*
  %sub_a = alloca %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix", align 8
  %sub_a.ascast = addrspacecast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix"* %sub_a to %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*
  %agg.tmp = alloca %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group", align 1
  %agg.tmp.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)*
  %sub_b = alloca %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix", align 8
  %sub_b.ascast = addrspacecast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix"* %sub_b to %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*
  %agg.tmp3 = alloca %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group", align 1
  %agg.tmp3.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp3 to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)*
  %sub_c = alloca %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix", align 8
  %sub_c.ascast = addrspacecast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix"* %sub_c to %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*
  %agg.tmp4 = alloca %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group", align 1
  %agg.tmp4.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp4 to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)*
  %cleanup.dest.slot = alloca i32, align 4
  %cleanup.dest.slot.ascast = addrspacecast i32* %cleanup.dest.slot to i32 addrspace(4)*
  %agg.tmp6 = alloca %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group", align 1
  %agg.tmp6.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp6 to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)*
  %agg.tmp7 = alloca %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", align 8
  %agg.tmp7.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %agg.tmp7 to %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*
  %ref.tmp = alloca %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", align 8
  %ref.tmp.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp to %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*
  %ref.tmp8 = alloca %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", align 8
  %ref.tmp8.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp8 to %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*
  %k = alloca i32, align 4
  %k.ascast = addrspacecast i32* %k to i32 addrspace(4)*
  %agg.tmp11 = alloca %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group", align 1
  %agg.tmp11.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp11 to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)*
  %agg.tmp12 = alloca %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", align 8
  %agg.tmp12.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %agg.tmp12 to %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*
  %ref.tmp13 = alloca %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", align 8
  %ref.tmp13.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp13 to %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*
  %ref.tmp14 = alloca %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", align 8
  %ref.tmp14.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp14 to %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*
  %agg.tmp19 = alloca %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group", align 1
  %agg.tmp19.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp19 to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)*
  %agg.tmp20 = alloca %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", align 8
  %agg.tmp20.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %agg.tmp20 to %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*
  %ref.tmp21 = alloca %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", align 8
  %ref.tmp21.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp21 to %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*
  %ref.tmp22 = alloca %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", align 8
  %ref.tmp22.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp22 to %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*
  %ref.tmp31 = alloca %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix", align 8
  %ref.tmp31.ascast = addrspacecast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix"* %ref.tmp31 to %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*
  %agg.tmp32 = alloca %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group", align 1
  %agg.tmp32.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp32 to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)*
  %agg.tmp33 = alloca %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group", align 1
  %agg.tmp33.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp33 to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)*
  %agg.tmp34 = alloca %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", align 8
  %agg.tmp34.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %agg.tmp34 to %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*
  %ref.tmp35 = alloca %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", align 8
  %ref.tmp35.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp35 to %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*
  %ref.tmp36 = alloca %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", align 8
  %ref.tmp36.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp36 to %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*
  store %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %this, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %spmd_item.ascast = addrspacecast %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item"* %spmd_item to %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)*
  %this1 = load %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)*, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = bitcast i64* %global_idx to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #12
  %retval.ascast.i = addrspacecast i64* %retval.i to i64 addrspace(4)*
  %this.addr.ascast.i = addrspacecast %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)** %this.addr.i to %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* addrspace(4)*
  %dimension.addr.ascast.i = addrspacecast i32* %dimension.addr.i to i32 addrspace(4)*
  %Id.ascast.i = addrspacecast i64* %Id.i to i64 addrspace(4)*
  store %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* %spmd_item.ascast, %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* addrspace(4)* %this.addr.ascast.i, align 8, !tbaa !7
  store i32 0, i32 addrspace(4)* %dimension.addr.ascast.i, align 4, !tbaa !23
  %this1.i = load %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)*, %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* addrspace(4)* %this.addr.ascast.i, align 8
  %1 = bitcast i64* %Id.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %1) #12
  %globalItem.i = getelementptr inbounds %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item", %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* %this1.i, i32 0, i32 0
  %2 = load i32, i32 addrspace(4)* %dimension.addr.ascast.i, align 4, !tbaa !23
  %retval.ascast.i.i = addrspacecast i64* %retval.i.i to i64 addrspace(4)*
  %this.addr.ascast.i.i = addrspacecast %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)** %this.addr.i.i to %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* addrspace(4)*
  %dimension.addr.ascast.i.i = addrspacecast i32* %dimension.addr.i.i to i32 addrspace(4)*
  %Id.ascast.i.i = addrspacecast i64* %Id.i.i to i64 addrspace(4)*
  store %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* %globalItem.i, %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* addrspace(4)* %this.addr.ascast.i.i, align 8, !tbaa !7
  store i32 %2, i32 addrspace(4)* %dimension.addr.ascast.i.i, align 4, !tbaa !23
  %this1.i.i = load %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)*, %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* addrspace(4)* %this.addr.ascast.i.i, align 8
  %3 = bitcast i64* %Id.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %3) #12
  %MImpl.i.i = getelementptr inbounds %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item", %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* %this1.i.i, i32 0, i32 0
  %MIndex.i.i = getelementptr inbounds %"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb1EEE.cl::sycl::detail::ItemBase", %"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb1EEE.cl::sycl::detail::ItemBase" addrspace(4)* %MImpl.i.i, i32 0, i32 1
  %4 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %MIndex.i.i to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %5 = load i32, i32 addrspace(4)* %dimension.addr.ascast.i.i, align 4, !tbaa !23
  %call.i.i = call spir_func i64 @_ZNK2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %4, i32 %5) #13
  store i64 %call.i.i, i64 addrspace(4)* %Id.ascast.i.i, align 8, !tbaa !11
  %6 = load i64, i64 addrspace(4)* %Id.ascast.i.i, align 8, !tbaa !11
  %cmp.i.i = icmp ule i64 %6, 2147483647
  call void @llvm.assume(i1 %cmp.i.i)
  %7 = load i64, i64 addrspace(4)* %Id.ascast.i.i, align 8, !tbaa !11
  %8 = bitcast i64* %Id.i.i to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %8) #12
  store i64 %7, i64 addrspace(4)* %Id.ascast.i, align 8, !tbaa !11
  %9 = load i64, i64 addrspace(4)* %Id.ascast.i, align 8, !tbaa !11
  %cmp.i = icmp ule i64 %9, 2147483647
  call void @llvm.assume(i1 %cmp.i)
  %10 = load i64, i64 addrspace(4)* %Id.ascast.i, align 8, !tbaa !11
  %11 = bitcast i64* %Id.i to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %11) #12
  store i64 %10, i64 addrspace(4)* %global_idx.ascast, align 8, !tbaa !11
  %12 = bitcast i64* %global_idy to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %12) #12
  %retval.ascast.i52 = addrspacecast i64* %retval.i51 to i64 addrspace(4)*
  %this.addr.ascast.i54 = addrspacecast %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)** %this.addr.i53 to %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* addrspace(4)*
  %dimension.addr.ascast.i56 = addrspacecast i32* %dimension.addr.i55 to i32 addrspace(4)*
  %Id.ascast.i58 = addrspacecast i64* %Id.i57 to i64 addrspace(4)*
  store %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* %spmd_item.ascast, %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* addrspace(4)* %this.addr.ascast.i54, align 8, !tbaa !7
  store i32 1, i32 addrspace(4)* %dimension.addr.ascast.i56, align 4, !tbaa !23
  %this1.i59 = load %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)*, %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* addrspace(4)* %this.addr.ascast.i54, align 8
  %13 = bitcast i64* %Id.i57 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %13) #12
  %globalItem.i60 = getelementptr inbounds %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item", %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* %this1.i59, i32 0, i32 0
  %14 = load i32, i32 addrspace(4)* %dimension.addr.ascast.i56, align 4, !tbaa !23
  %retval.ascast.i.i61 = addrspacecast i64* %retval.i.i47 to i64 addrspace(4)*
  %this.addr.ascast.i.i62 = addrspacecast %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)** %this.addr.i.i48 to %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* addrspace(4)*
  %dimension.addr.ascast.i.i63 = addrspacecast i32* %dimension.addr.i.i49 to i32 addrspace(4)*
  %Id.ascast.i.i64 = addrspacecast i64* %Id.i.i50 to i64 addrspace(4)*
  store %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* %globalItem.i60, %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* addrspace(4)* %this.addr.ascast.i.i62, align 8, !tbaa !7
  store i32 %14, i32 addrspace(4)* %dimension.addr.ascast.i.i63, align 4, !tbaa !23
  %this1.i.i65 = load %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)*, %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* addrspace(4)* %this.addr.ascast.i.i62, align 8
  %15 = bitcast i64* %Id.i.i50 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %15) #12
  %MImpl.i.i66 = getelementptr inbounds %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item", %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* %this1.i.i65, i32 0, i32 0
  %MIndex.i.i67 = getelementptr inbounds %"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb1EEE.cl::sycl::detail::ItemBase", %"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb1EEE.cl::sycl::detail::ItemBase" addrspace(4)* %MImpl.i.i66, i32 0, i32 1
  %16 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %MIndex.i.i67 to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %17 = load i32, i32 addrspace(4)* %dimension.addr.ascast.i.i63, align 4, !tbaa !23
  %call.i.i68 = call spir_func i64 @_ZNK2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %16, i32 %17) #13
  store i64 %call.i.i68, i64 addrspace(4)* %Id.ascast.i.i64, align 8, !tbaa !11
  %18 = load i64, i64 addrspace(4)* %Id.ascast.i.i64, align 8, !tbaa !11
  %cmp.i.i69 = icmp ule i64 %18, 2147483647
  call void @llvm.assume(i1 %cmp.i.i69)
  %19 = load i64, i64 addrspace(4)* %Id.ascast.i.i64, align 8, !tbaa !11
  %20 = bitcast i64* %Id.i.i50 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %20) #12
  store i64 %19, i64 addrspace(4)* %Id.ascast.i58, align 8, !tbaa !11
  %21 = load i64, i64 addrspace(4)* %Id.ascast.i58, align 8, !tbaa !11
  %cmp.i70 = icmp ule i64 %21, 2147483647
  call void @llvm.assume(i1 %cmp.i70)
  %22 = load i64, i64 addrspace(4)* %Id.ascast.i58, align 8, !tbaa !11
  %23 = bitcast i64* %Id.i57 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %23) #12
  store i64 %22, i64 addrspace(4)* %global_idy.ascast, align 8, !tbaa !11
  %24 = bitcast i64* %sg_startx to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %24) #12
  %25 = load i64, i64 addrspace(4)* %global_idx.ascast, align 8, !tbaa !11
  store i64 %25, i64 addrspace(4)* %sg_startx.ascast, align 8, !tbaa !11
  %26 = bitcast i64* %sg_starty to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %26) #12
  %27 = load i64, i64 addrspace(4)* %global_idy.ascast, align 8, !tbaa !11
  store i64 %27, i64 addrspace(4)* %sg_starty.ascast, align 8, !tbaa !11
  %28 = bitcast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %sg to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %28) #12
  call spir_func void @_ZNK2cl4sycl7nd_itemILi2EE13get_sub_groupEv(%"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)* sret(%"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group") align 1 %sg.ascast, %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* align 8 dereferenceable_or_null(144) %spmd_item.ascast) #13
  %29 = bitcast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix"* %sub_a to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %29) #12
  %agg.tmp.ascast.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)* %agg.tmp.ascast to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"*
  call spir_func void @_ZN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEC2ES7_(%"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* align 8 dereferenceable_or_null(8) %sub_a.ascast, %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* byval(%"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group") align 1 %agg.tmp.ascast.ascast) #13
  %30 = bitcast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix"* %sub_b to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %30) #12
  %agg.tmp3.ascast.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)* %agg.tmp3.ascast to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"*
  call spir_func void @_ZN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEC2ES7_(%"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* align 8 dereferenceable_or_null(8) %sub_b.ascast, %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* byval(%"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group") align 1 %agg.tmp3.ascast.ascast) #13
  %31 = bitcast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix"* %sub_c to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %31) #12
  %agg.tmp4.ascast.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)* %agg.tmp4.ascast to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"*
  call spir_func void @_ZN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEC2ES7_(%"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* align 8 dereferenceable_or_null(8) %sub_c.ascast, %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* byval(%"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group") align 1 %agg.tmp4.ascast.ascast) #13
  %retval.ascast.i76 = addrspacecast i64* %retval.i75 to i64 addrspace(4)*
  %this.addr.ascast.i78 = addrspacecast %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)** %this.addr.i77 to %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* addrspace(4)*
  %dimension.addr.ascast.i80 = addrspacecast i32* %dimension.addr.i79 to i32 addrspace(4)*
  %Id.ascast.i82 = addrspacecast i64* %Id.i81 to i64 addrspace(4)*
  store %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* %spmd_item.ascast, %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* addrspace(4)* %this.addr.ascast.i78, align 8, !tbaa !7
  store i32 1, i32 addrspace(4)* %dimension.addr.ascast.i80, align 4, !tbaa !23
  %this1.i83 = load %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)*, %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* addrspace(4)* %this.addr.ascast.i78, align 8
  %32 = bitcast i64* %Id.i81 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %32) #12
  %localItem.i = getelementptr inbounds %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item", %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* %this1.i83, i32 0, i32 1
  %33 = load i32, i32 addrspace(4)* %dimension.addr.ascast.i80, align 4, !tbaa !23
  %retval.ascast.i.i84 = addrspacecast i64* %retval.i.i71 to i64 addrspace(4)*
  %this.addr.ascast.i.i85 = addrspacecast %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)** %this.addr.i.i72 to %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* addrspace(4)*
  %dimension.addr.ascast.i.i86 = addrspacecast i32* %dimension.addr.i.i73 to i32 addrspace(4)*
  %Id.ascast.i.i87 = addrspacecast i64* %Id.i.i74 to i64 addrspace(4)*
  store %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* %localItem.i, %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* addrspace(4)* %this.addr.ascast.i.i85, align 8, !tbaa !7
  store i32 %33, i32 addrspace(4)* %dimension.addr.ascast.i.i86, align 4, !tbaa !23
  %this1.i.i88 = load %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)*, %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* addrspace(4)* %this.addr.ascast.i.i85, align 8
  %34 = bitcast i64* %Id.i.i74 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %34) #12
  %MImpl.i.i89 = getelementptr inbounds %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item", %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* %this1.i.i88, i32 0, i32 0
  %MIndex.i.i90 = getelementptr inbounds %"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb0EEE.cl::sycl::detail::ItemBase", %"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb0EEE.cl::sycl::detail::ItemBase" addrspace(4)* %MImpl.i.i89, i32 0, i32 1
  %35 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %MIndex.i.i90 to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %36 = load i32, i32 addrspace(4)* %dimension.addr.ascast.i.i86, align 4, !tbaa !23
  %call.i.i91 = call spir_func i64 @_ZNK2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %35, i32 %36) #14
  store i64 %call.i.i91, i64 addrspace(4)* %Id.ascast.i.i87, align 8, !tbaa !11
  %37 = load i64, i64 addrspace(4)* %Id.ascast.i.i87, align 8, !tbaa !11
  %cmp.i.i92 = icmp ule i64 %37, 2147483647
  call void @llvm.assume(i1 %cmp.i.i92) #12
  %38 = load i64, i64 addrspace(4)* %Id.ascast.i.i87, align 8, !tbaa !11
  %39 = bitcast i64* %Id.i.i74 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %39) #12
  store i64 %38, i64 addrspace(4)* %Id.ascast.i82, align 8, !tbaa !11
  %40 = load i64, i64 addrspace(4)* %Id.ascast.i82, align 8, !tbaa !11
  %cmp.i93 = icmp ule i64 %40, 2147483647
  call void @llvm.assume(i1 %cmp.i93)
  %41 = load i64, i64 addrspace(4)* %Id.ascast.i82, align 8, !tbaa !11
  %42 = bitcast i64* %Id.i81 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %42) #12
  %rem = urem i64 %41, 16
  %tobool = icmp ne i64 %rem, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 1, i32 addrspace(4)* %cleanup.dest.slot.ascast, align 4
  br label %cleanup

if.end:                                           ; preds = %entry
  %43 = bitcast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %43) #12
  %44 = bitcast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp8 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %44) #12
  %45 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %this1, i32 0, i32 2
  call spir_func void @_ZNK2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE11get_pointerILS4_2014EvEENS0_9multi_ptrIfLNS2_13address_spaceE1EEEv(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* sret(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr") align 8 %ref.tmp8.ascast, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %45) #13
  %46 = load i64, i64 addrspace(4)* %sg_startx.ascast, align 8, !tbaa !11
  %mul = mul i64 %46, 47
  %47 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %this1, i32 0, i32 4
  %48 = load i64, i64 addrspace(4)* %47, align 8, !tbaa !20
  %mul9 = mul i64 %mul, %48
  call spir_func void @_ZNK2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEplEl(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* sret(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr") align 8 %ref.tmp.ascast, %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %ref.tmp8.ascast, i64 %mul9) #13
  %49 = load i64, i64 addrspace(4)* %sg_starty.ascast, align 8, !tbaa !11
  %mul10 = mul i64 %49, 47
  call spir_func void @_ZNK2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEplEl(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* sret(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr") align 8 %agg.tmp7.ascast, %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %ref.tmp.ascast, i64 %mul10) #13
  %50 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %this1, i32 0, i32 4
  %51 = load i64, i64 addrspace(4)* %50, align 8, !tbaa !20
  %agg.tmp6.ascast.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)* %agg.tmp6.ascast to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"*
  %agg.tmp7.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* %agg.tmp7.ascast to %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"*
  %52 = bitcast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %agg.tmp7.ascast.ascast95 to i8*
  %53 = bitcast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %agg.tmp7.ascast.ascast to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %52, i8* align 1 %53, i64 8, i1 false)
  %54 = bitcast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp6.ascast.ascast94 to i8*
  %55 = bitcast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp6.ascast.ascast to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %54, i8* align 1 %55, i64 1, i1 false)
  %res.addr.ascast.i = addrspacecast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)** %res.addr.i to %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)*
  %stride.addr.ascast.i = addrspacecast i64* %stride.addr.i to i64 addrspace(4)*
  %L.addr.ascast.i = addrspacecast i32* %L.addr.i to i32 addrspace(4)*
  %Ptr.ascast.i = addrspacecast float addrspace(4)** %Ptr.i to float addrspace(4)* addrspace(4)*
  %sg.ascast.i = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp6.ascast.ascast94 to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)*
  store %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %sub_c.ascast, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %res.addr.ascast.i, align 8, !tbaa !7
  %src.ascast.i = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %agg.tmp7.ascast.ascast95 to %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*
  store i64 %51, i64 addrspace(4)* %stride.addr.ascast.i, align 8, !tbaa !11
  store i32 0, i32 addrspace(4)* %L.addr.ascast.i, align 4, !tbaa !29
  %56 = bitcast float addrspace(4)** %Ptr.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %56) #12
  %call.i = call spir_func float addrspace(1)* @_ZNK2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EE3getEv(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %src.ascast.i) #13
  %call.ascast.i = addrspacecast float addrspace(1)* %call.i to float addrspace(4)*
  store float addrspace(4)* %call.ascast.i, float addrspace(4)* addrspace(4)* %Ptr.ascast.i, align 8, !tbaa !7
  %57 = load float addrspace(4)*, float addrspace(4)* addrspace(4)* %Ptr.ascast.i, align 8, !tbaa !7
  %58 = load i64, i64 addrspace(4)* %stride.addr.ascast.i, align 8, !tbaa !11
  %call1.i = call spir_func %"struct._ZTSN5__spv19__spirv_MatrixINTELIfLm47ELm47ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* @_Z23__spirv_MatrixLoadINTELIfLm47ELm47ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEPNS0_19__spirv_MatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPS5_mS1_S3_i(float addrspace(4)* %57, i64 %58, i32 0, i32 3, i32 0) #13
  %59 = load %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %res.addr.ascast.i, align 8, !tbaa !7
  %spvm.i = getelementptr inbounds %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix", %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %59, i32 0, i32 0
  store %"struct._ZTSN5__spv19__spirv_MatrixINTELIfLm47ELm47ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* %call1.i, %"struct._ZTSN5__spv19__spirv_MatrixINTELIfLm47ELm47ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* addrspace(4)* %spvm.i, align 8, !tbaa !31
  %60 = bitcast float addrspace(4)** %Ptr.i to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %60) #12
  %61 = bitcast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp8 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %61) #12
  %62 = bitcast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %62) #12
  %63 = bitcast i32* %k to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %63) #12
  store i32 0, i32 addrspace(4)* %k.ascast, align 4, !tbaa !23
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %if.end
  %64 = load i32, i32 addrspace(4)* %k.ascast, align 4, !tbaa !23
  %conv = sext i32 %64 to i64
  %65 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %this1, i32 0, i32 5
  %66 = load i64, i64 addrspace(4)* %65, align 8, !tbaa !21
  %div = udiv i64 %66, 146
  %cmp = icmp ult i64 %conv, %div
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  store i32 2, i32 addrspace(4)* %cleanup.dest.slot.ascast, align 4
  %67 = bitcast i32* %k to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %67) #12
  br label %for.end

for.body:                                         ; preds = %for.cond
  %68 = bitcast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %68) #12
  %69 = bitcast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %69) #12
  %70 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %this1, i32 0, i32 0
  call spir_func void @_ZNK2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE11get_pointerILS4_2014EvEENS0_9multi_ptrItLNS2_13address_spaceE1EEEv(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* sret(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr") align 8 %ref.tmp14.ascast, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %70) #13
  %71 = load i64, i64 addrspace(4)* %sg_startx.ascast, align 8, !tbaa !11
  %mul15 = mul i64 %71, 47
  %72 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %this1, i32 0, i32 5
  %73 = load i64, i64 addrspace(4)* %72, align 8, !tbaa !21
  %mul16 = mul i64 %mul15, %73
  call spir_func void @_ZNK2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEplEl(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* sret(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr") align 8 %ref.tmp13.ascast, %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %ref.tmp14.ascast, i64 %mul16) #13
  %74 = load i32, i32 addrspace(4)* %k.ascast, align 4, !tbaa !23
  %mul17 = mul nsw i32 %74, 146
  %conv18 = sext i32 %mul17 to i64
  call spir_func void @_ZNK2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEplEl(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* sret(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr") align 8 %agg.tmp12.ascast, %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %ref.tmp13.ascast, i64 %conv18) #13
  %75 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %this1, i32 0, i32 5
  %76 = load i64, i64 addrspace(4)* %75, align 8, !tbaa !21
  %agg.tmp11.ascast.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)* %agg.tmp11.ascast to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"*
  %agg.tmp12.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* %agg.tmp12.ascast to %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"*
  %77 = bitcast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %agg.tmp12.ascast.ascast97 to i8*
  %78 = bitcast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %agg.tmp12.ascast.ascast to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %77, i8* align 1 %78, i64 8, i1 false)
  %79 = bitcast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp11.ascast.ascast96 to i8*
  %80 = bitcast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp11.ascast.ascast to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %79, i8* align 1 %80, i64 1, i1 false)
  %res.addr.ascast.i99 = addrspacecast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)** %res.addr.i98 to %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)*
  %stride.addr.ascast.i101 = addrspacecast i64* %stride.addr.i100 to i64 addrspace(4)*
  %L.addr.ascast.i103 = addrspacecast i32* %L.addr.i102 to i32 addrspace(4)*
  %Ptr.ascast.i105 = addrspacecast i16 addrspace(4)** %Ptr.i104 to i16 addrspace(4)* addrspace(4)*
  %sg.ascast.i106 = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp11.ascast.ascast96 to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)*
  store %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %sub_a.ascast, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %res.addr.ascast.i99, align 8, !tbaa !7
  %src.ascast.i107 = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %agg.tmp12.ascast.ascast97 to %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*
  store i64 %76, i64 addrspace(4)* %stride.addr.ascast.i101, align 8, !tbaa !11
  store i32 0, i32 addrspace(4)* %L.addr.ascast.i103, align 4, !tbaa !29
  %81 = bitcast i16 addrspace(4)** %Ptr.i104 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %81) #12
  %call.i108 = call spir_func i16 addrspace(1)* @_ZNK2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EE3getEv(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %src.ascast.i107) #13
  %call.ascast.i109 = addrspacecast i16 addrspace(1)* %call.i108 to i16 addrspace(4)*
  store i16 addrspace(4)* %call.ascast.i109, i16 addrspace(4)* addrspace(4)* %Ptr.ascast.i105, align 8, !tbaa !7
  %82 = load i16 addrspace(4)*, i16 addrspace(4)* addrspace(4)* %Ptr.ascast.i105, align 8, !tbaa !7
  %83 = load i64, i64 addrspace(4)* %stride.addr.ascast.i101, align 8, !tbaa !11
  %call1.i110 = call spir_func %"struct._ZTSN5__spv19__spirv_MatrixINTELItLm47ELm146ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* @_Z23__spirv_MatrixLoadINTELItLm47ELm146ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEPNS0_19__spirv_MatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPS5_mS1_S3_i(i16 addrspace(4)* %82, i64 %83, i32 0, i32 3, i32 0) #13
  %84 = load %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %res.addr.ascast.i99, align 8, !tbaa !7
  %spvm.i111 = getelementptr inbounds %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix", %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %84, i32 0, i32 0
  store %"struct._ZTSN5__spv19__spirv_MatrixINTELItLm47ELm146ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* %call1.i110, %"struct._ZTSN5__spv19__spirv_MatrixINTELItLm47ELm146ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* addrspace(4)* %spvm.i111, align 8, !tbaa !33
  %85 = bitcast i16 addrspace(4)** %Ptr.i104 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %85) #12
  %86 = bitcast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp14 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %86) #12
  %87 = bitcast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp13 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %87) #12
  %88 = bitcast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %88) #12
  %89 = bitcast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %89) #12
  %90 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %this1, i32 0, i32 1
  call spir_func void @_ZNK2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE11get_pointerILS4_2014EvEENS0_9multi_ptrItLNS2_13address_spaceE1EEEv(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* sret(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr") align 8 %ref.tmp22.ascast, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %90) #13
  %91 = load i32, i32 addrspace(4)* %k.ascast, align 4, !tbaa !23
  %mul23 = mul nsw i32 %91, 146
  %div24 = sdiv i32 %mul23, 2
  %conv25 = sext i32 %div24 to i64
  %92 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %this1, i32 0, i32 4
  %93 = load i64, i64 addrspace(4)* %92, align 8, !tbaa !20
  %mul26 = mul i64 %93, 2
  %mul27 = mul i64 %conv25, %mul26
  call spir_func void @_ZNK2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEplEl(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* sret(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr") align 8 %ref.tmp21.ascast, %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %ref.tmp22.ascast, i64 %mul27) #13
  %94 = load i64, i64 addrspace(4)* %sg_starty.ascast, align 8, !tbaa !11
  %mul28 = mul i64 %94, 47
  %mul29 = mul i64 %mul28, 2
  call spir_func void @_ZNK2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEplEl(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* sret(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr") align 8 %agg.tmp20.ascast, %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %ref.tmp21.ascast, i64 %mul29) #13
  %95 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %this1, i32 0, i32 4
  %96 = load i64, i64 addrspace(4)* %95, align 8, !tbaa !20
  %mul30 = mul i64 %96, 2
  %agg.tmp19.ascast.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)* %agg.tmp19.ascast to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"*
  %agg.tmp20.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* %agg.tmp20.ascast to %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"*
  %97 = bitcast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %agg.tmp20.ascast.ascast113 to i8*
  %98 = bitcast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %agg.tmp20.ascast.ascast to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %97, i8* align 1 %98, i64 8, i1 false)
  %99 = bitcast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp19.ascast.ascast112 to i8*
  %100 = bitcast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp19.ascast.ascast to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %99, i8* align 1 %100, i64 1, i1 false)
  %res.addr.ascast.i115 = addrspacecast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)** %res.addr.i114 to %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)*
  %stride.addr.ascast.i117 = addrspacecast i64* %stride.addr.i116 to i64 addrspace(4)*
  %L.addr.ascast.i119 = addrspacecast i32* %L.addr.i118 to i32 addrspace(4)*
  %Ptr.ascast.i121 = addrspacecast i16 addrspace(4)** %Ptr.i120 to i16 addrspace(4)* addrspace(4)*
  %sg.ascast.i122 = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp19.ascast.ascast112 to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)*
  store %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %sub_b.ascast, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %res.addr.ascast.i115, align 8, !tbaa !7
  %src.ascast.i123 = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %agg.tmp20.ascast.ascast113 to %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*
  store i64 %mul30, i64 addrspace(4)* %stride.addr.ascast.i117, align 8, !tbaa !11
  store i32 3, i32 addrspace(4)* %L.addr.ascast.i119, align 4, !tbaa !29
  %101 = bitcast i16 addrspace(4)** %Ptr.i120 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %101) #12
  %call.i124 = call spir_func i16 addrspace(1)* @_ZNK2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EE3getEv(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %src.ascast.i123) #13
  %call.ascast.i125 = addrspacecast i16 addrspace(1)* %call.i124 to i16 addrspace(4)*
  store i16 addrspace(4)* %call.ascast.i125, i16 addrspace(4)* addrspace(4)* %Ptr.ascast.i121, align 8, !tbaa !7
  %102 = load i16 addrspace(4)*, i16 addrspace(4)* addrspace(4)* %Ptr.ascast.i121, align 8, !tbaa !7
  %103 = load i64, i64 addrspace(4)* %stride.addr.ascast.i117, align 8, !tbaa !11
  %call1.i126 = call spir_func %"struct._ZTSN5__spv19__spirv_MatrixINTELItLm146ELm47ELNS_12MatrixLayoutE3ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* @_Z23__spirv_MatrixLoadINTELItLm146ELm47ELN5__spv12MatrixLayoutE3ELNS0_5Scope4FlagE3EEPNS0_19__spirv_MatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPS5_mS1_S3_i(i16 addrspace(4)* %102, i64 %103, i32 3, i32 3, i32 0) #13
  %104 = load %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %res.addr.ascast.i115, align 8, !tbaa !7
  %spvm.i127 = getelementptr inbounds %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix", %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %104, i32 0, i32 0
  store %"struct._ZTSN5__spv19__spirv_MatrixINTELItLm146ELm47ELNS_12MatrixLayoutE3ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* %call1.i126, %"struct._ZTSN5__spv19__spirv_MatrixINTELItLm146ELm47ELNS_12MatrixLayoutE3ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* addrspace(4)* %spvm.i127, align 8, !tbaa !35
  %105 = bitcast i16 addrspace(4)** %Ptr.i120 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %105) #12
  %106 = bitcast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp22 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %106) #12
  %107 = bitcast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp21 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %107) #12
  %108 = bitcast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix"* %ref.tmp31 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %108) #12
  %agg.tmp32.ascast.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)* %agg.tmp32.ascast to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"*
  call void @llvm.experimental.noalias.scope.decl(metadata !37)
  %109 = bitcast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp32.ascast.ascast128 to i8*
  %110 = bitcast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp32.ascast.ascast to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %109, i8* align 1 %110, i64 1, i1 false)
  %mA.addr.ascast.i = addrspacecast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)** %mA.addr.i to %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)*
  %mB.addr.ascast.i = addrspacecast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)** %mB.addr.i to %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)*
  %mC.addr.ascast.i = addrspacecast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)** %mC.addr.i to %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)*
  %agg.tmp.ascast.i = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp.i to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)*
  %sg.ascast.i129 = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp32.ascast.ascast128 to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)*
  store %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %sub_a.ascast, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %mA.addr.ascast.i, align 8, !tbaa !7, !noalias !37
  store %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %sub_b.ascast, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %mB.addr.ascast.i, align 8, !tbaa !7, !noalias !37
  store %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %sub_c.ascast, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %mC.addr.ascast.i, align 8, !tbaa !7, !noalias !37
  call spir_func void @_ZN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEC2ES7_(%"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* align 8 dereferenceable_or_null(8) %ref.tmp31.ascast, %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* byval(%"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group") align 1 %agg.tmp.i) #13
  %111 = load %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %mA.addr.ascast.i, align 8, !tbaa !7, !noalias !37
  %spvm.i130 = getelementptr inbounds %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix", %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %111, i32 0, i32 0
  %112 = load %"struct._ZTSN5__spv19__spirv_MatrixINTELItLm47ELm146ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)*, %"struct._ZTSN5__spv19__spirv_MatrixINTELItLm47ELm146ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* addrspace(4)* %spvm.i130, align 8, !tbaa !33
  %113 = load %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %mB.addr.ascast.i, align 8, !tbaa !7, !noalias !37
  %spvm1.i = getelementptr inbounds %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix", %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %113, i32 0, i32 0
  %114 = load %"struct._ZTSN5__spv19__spirv_MatrixINTELItLm146ELm47ELNS_12MatrixLayoutE3ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)*, %"struct._ZTSN5__spv19__spirv_MatrixINTELItLm146ELm47ELNS_12MatrixLayoutE3ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* addrspace(4)* %spvm1.i, align 8, !tbaa !35
  %115 = load %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %mC.addr.ascast.i, align 8, !tbaa !7, !noalias !37
  %spvm2.i = getelementptr inbounds %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix", %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %115, i32 0, i32 0
  %116 = load %"struct._ZTSN5__spv19__spirv_MatrixINTELIfLm47ELm47ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)*, %"struct._ZTSN5__spv19__spirv_MatrixINTELIfLm47ELm47ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* addrspace(4)* %spvm2.i, align 8, !tbaa !31
  %call.i131 = call spir_func %"struct._ZTSN5__spv19__spirv_MatrixINTELIfLm47ELm47ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* @_Z22__spirv_MatrixMadINTELItfLm47ELm146ELm47ELN5__spv12MatrixLayoutE0ELS1_3ELS1_0ELNS0_5Scope4FlagE3EEPNS0_19__spirv_MatrixINTELIT0_XT1_EXT3_EXT6_EXT7_EEEPNS4_IT_XT1_EXT2_EXT4_EXT7_EEEPNS4_IS8_XT2_EXT3_EXT5_EXT7_EEES7_S3_(%"struct._ZTSN5__spv19__spirv_MatrixINTELItLm47ELm146ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* %112, %"struct._ZTSN5__spv19__spirv_MatrixINTELItLm146ELm47ELNS_12MatrixLayoutE3ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* %114, %"struct._ZTSN5__spv19__spirv_MatrixINTELIfLm47ELm47ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* %116, i32 3) #13
  %spvm3.i = getelementptr inbounds %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix", %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %ref.tmp31.ascast, i32 0, i32 0
  store %"struct._ZTSN5__spv19__spirv_MatrixINTELIfLm47ELm47ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* %call.i131, %"struct._ZTSN5__spv19__spirv_MatrixINTELIfLm47ELm47ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* addrspace(4)* %spvm3.i, align 8, !tbaa !31, !alias.scope !37
  %117 = bitcast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %sub_c.ascast to i8 addrspace(4)*
  %118 = bitcast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %ref.tmp31.ascast to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %117, i8 addrspace(4)* align 8 %118, i64 8, i1 false), !tbaa.struct !40
  %119 = bitcast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix"* %ref.tmp31 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %119) #12
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %120 = load i32, i32 addrspace(4)* %k.ascast, align 4, !tbaa !23
  %add = add nsw i32 %120, 1
  store i32 %add, i32 addrspace(4)* %k.ascast, align 4, !tbaa !23
  br label %for.cond, !llvm.loop !41

for.end:                                          ; preds = %for.cond.cleanup
  %121 = bitcast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp35 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %121) #12
  %122 = bitcast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp36 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %122) #12
  %123 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %this1, i32 0, i32 2
  call spir_func void @_ZNK2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE11get_pointerILS4_2014EvEENS0_9multi_ptrIfLNS2_13address_spaceE1EEEv(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* sret(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr") align 8 %ref.tmp36.ascast, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %123) #13
  %124 = load i64, i64 addrspace(4)* %sg_startx.ascast, align 8, !tbaa !11
  %mul37 = mul i64 %124, 47
  %125 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %this1, i32 0, i32 4
  %126 = load i64, i64 addrspace(4)* %125, align 8, !tbaa !20
  %mul38 = mul i64 %mul37, %126
  call spir_func void @_ZNK2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEplEl(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* sret(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr") align 8 %ref.tmp35.ascast, %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %ref.tmp36.ascast, i64 %mul38) #13
  %127 = load i64, i64 addrspace(4)* %sg_starty.ascast, align 8, !tbaa !11
  %mul39 = mul i64 %127, 47
  call spir_func void @_ZNK2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEplEl(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* sret(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr") align 8 %agg.tmp34.ascast, %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %ref.tmp35.ascast, i64 %mul39) #13
  %128 = getelementptr inbounds %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon, %class._ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_.anon addrspace(4)* %this1, i32 0, i32 4
  %129 = load i64, i64 addrspace(4)* %128, align 8, !tbaa !20
  %agg.tmp33.ascast.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)* %agg.tmp33.ascast to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"*
  %agg.tmp34.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* %agg.tmp34.ascast to %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"*
  %130 = bitcast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %agg.tmp34.ascast.ascast133 to i8*
  %131 = bitcast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %agg.tmp34.ascast.ascast to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %130, i8* align 1 %131, i64 8, i1 false)
  %132 = bitcast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp33.ascast.ascast132 to i8*
  %133 = bitcast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp33.ascast.ascast to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %132, i8* align 1 %133, i64 1, i1 false)
  %res.addr.ascast.i135 = addrspacecast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)** %res.addr.i134 to %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)*
  %stride.addr.ascast.i137 = addrspacecast i64* %stride.addr.i136 to i64 addrspace(4)*
  %L.addr.ascast.i139 = addrspacecast i32* %L.addr.i138 to i32 addrspace(4)*
  %Ptr.ascast.i141 = addrspacecast float addrspace(4)** %Ptr.i140 to float addrspace(4)* addrspace(4)*
  %sg.ascast.i142 = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %agg.tmp33.ascast.ascast132 to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)*
  store %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %sub_c.ascast, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %res.addr.ascast.i135, align 8, !tbaa !7
  %src.ascast.i143 = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %agg.tmp34.ascast.ascast133 to %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*
  store i64 %129, i64 addrspace(4)* %stride.addr.ascast.i137, align 8, !tbaa !11
  store i32 0, i32 addrspace(4)* %L.addr.ascast.i139, align 4, !tbaa !29
  %134 = bitcast float addrspace(4)** %Ptr.i140 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %134) #12
  %call.i144 = call spir_func float addrspace(1)* @_ZNK2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EE3getEv(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %src.ascast.i143) #13
  %call.ascast.i145 = addrspacecast float addrspace(1)* %call.i144 to float addrspace(4)*
  store float addrspace(4)* %call.ascast.i145, float addrspace(4)* addrspace(4)* %Ptr.ascast.i141, align 8, !tbaa !7
  %135 = load float addrspace(4)*, float addrspace(4)* addrspace(4)* %Ptr.ascast.i141, align 8, !tbaa !7
  %136 = load %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %res.addr.ascast.i135, align 8, !tbaa !7
  %spvm.i146 = getelementptr inbounds %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix", %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %136, i32 0, i32 0
  %137 = load %"struct._ZTSN5__spv19__spirv_MatrixINTELIfLm47ELm47ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)*, %"struct._ZTSN5__spv19__spirv_MatrixINTELIfLm47ELm47ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* addrspace(4)* %spvm.i146, align 8, !tbaa !31
  %138 = load i64, i64 addrspace(4)* %stride.addr.ascast.i137, align 8, !tbaa !11
  call spir_func void @_Z24__spirv_MatrixStoreINTELIfLm47ELm47ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEvPT_PNS0_19__spirv_MatrixINTELIS4_XT0_EXT1_EXT2_EXT3_EEEmS1_S3_i(float addrspace(4)* %135, %"struct._ZTSN5__spv19__spirv_MatrixINTELIfLm47ELm47ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* %137, i64 %138, i32 0, i32 3, i32 0) #13
  %139 = bitcast float addrspace(4)** %Ptr.i140 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %139) #12
  %140 = bitcast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp36 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %140) #12
  %141 = bitcast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr"* %ref.tmp35 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %141) #12
  store i32 0, i32 addrspace(4)* %cleanup.dest.slot.ascast, align 4
  br label %cleanup

cleanup:                                          ; preds = %for.end, %if.then
  %142 = bitcast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix"* %sub_c to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %142) #12
  %143 = bitcast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix"* %sub_b to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %143) #12
  %144 = bitcast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix"* %sub_a to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %144) #12
  %145 = bitcast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %sg to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %145) #12
  %146 = bitcast i64* %sg_starty to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %146) #12
  %147 = bitcast i64* %sg_startx to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %147) #12
  %148 = bitcast i64* %global_idy to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %148) #12
  %149 = bitcast i64* %global_idx to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %149) #12
  %cleanup.dest = load i32, i32 addrspace(4)* %cleanup.dest.slot.ascast, align 4
  switch i32 %cleanup.dest, label %unreachable [
    i32 0, label %cleanup.cont
    i32 1, label %cleanup.cont
  ]

cleanup.cont:                                     ; preds = %cleanup, %cleanup
  ret void

unreachable:                                      ; preds = %cleanup
  unreachable
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl6detail7Builder10getElementILi2EEEKNS0_7nd_itemIXT_EEEPS5_(%"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item") align 8 %agg.result, %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* %0) #3 comdat align 2 {
entry:
  %.addr = alloca %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)*, align 8
  %.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)** %.addr to %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* addrspace(4)*
  %GlobalSize = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", align 8
  %GlobalSize.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %GlobalSize to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %LocalSize = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", align 8
  %LocalSize.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %LocalSize to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %GroupRange = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", align 8
  %GroupRange.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %GroupRange to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %GroupId = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id", align 8
  %GroupId.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %GroupId to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*
  %GlobalId = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id", align 8
  %GlobalId.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %GlobalId to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*
  %LocalId = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id", align 8
  %LocalId.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %LocalId to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*
  %GlobalOffset = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id", align 8
  %GlobalOffset.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %GlobalOffset to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*
  %Group = alloca %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group", align 8
  %Group.ascast = addrspacecast %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group"* %Group to %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)*
  %GlobalItem = alloca %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item", align 8
  %GlobalItem.ascast = addrspacecast %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item"* %GlobalItem to %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)*
  %LocalItem = alloca %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item", align 8
  %LocalItem.ascast = addrspacecast %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item"* %LocalItem to %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)*
  %cleanup.dest.slot = alloca i32, align 4
  store %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* %0, %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* addrspace(4)* %.addr.ascast, align 8, !tbaa !7
  %1 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %GlobalSize to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %1) #12
  call spir_func void @_ZN7__spirvL14initGlobalSizeILi2EN2cl4sycl5rangeILi2EEEEET0_v(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* sret(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %GlobalSize.ascast) #13
  %2 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %LocalSize to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %2) #12
  call spir_func void @_ZN7__spirvL17initWorkgroupSizeILi2EN2cl4sycl5rangeILi2EEEEET0_v(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* sret(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %LocalSize.ascast) #13
  %3 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %GroupRange to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %3) #12
  call spir_func void @_ZN7__spirvL17initNumWorkgroupsILi2EN2cl4sycl5rangeILi2EEEEET0_v(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* sret(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %GroupRange.ascast) #13
  %4 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %GroupId to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %4) #12
  call spir_func void @_ZN7__spirvL15initWorkgroupIdILi2EN2cl4sycl2idILi2EEEEET0_v(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* sret(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %GroupId.ascast) #13
  %5 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %GlobalId to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %5) #12
  call spir_func void @_ZN7__spirvL22initGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEEET0_v(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* sret(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %GlobalId.ascast) #13
  %6 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %LocalId to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %6) #12
  call spir_func void @_ZN7__spirvL21initLocalInvocationIdILi2EN2cl4sycl2idILi2EEEEET0_v(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* sret(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %LocalId.ascast) #13
  %7 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %GlobalOffset to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %7) #12
  call spir_func void @_ZN7__spirvL16initGlobalOffsetILi2EN2cl4sycl2idILi2EEEEET0_v(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* sret(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %GlobalOffset.ascast) #13
  %8 = bitcast %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group"* %Group to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* %8) #12
  call spir_func void @_ZN2cl4sycl6detail7Builder11createGroupILi2EEENS0_5groupIXT_EEERKNS0_5rangeIXT_EEES9_S9_RKNS0_2idIXT_EEE(%"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* sret(%"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group") align 8 %Group.ascast, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable(16) %GlobalSize.ascast, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable(16) %LocalSize.ascast, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable(16) %GroupRange.ascast, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable(16) %GroupId.ascast) #13
  %9 = bitcast %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item"* %GlobalItem to i8*
  call void @llvm.lifetime.start.p0i8(i64 48, i8* %9) #12
  call spir_func void @_ZN2cl4sycl6detail7Builder10createItemILi2ELb1EEENSt9enable_ifIXT0_ENS0_4itemIXT_EXT0_EEEE4typeERKNS0_5rangeIXT_EEERKNS0_2idIXT_EEESG_(%"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* sret(%"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item") align 8 %GlobalItem.ascast, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable(16) %GlobalSize.ascast, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable(16) %GlobalId.ascast, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable(16) %GlobalOffset.ascast) #13
  %10 = bitcast %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item"* %LocalItem to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* %10) #12
  call spir_func void @_ZN2cl4sycl6detail7Builder10createItemILi2ELb0EEENSt9enable_ifIXntT0_ENS0_4itemIXT_EXT0_EEEE4typeERKNS0_5rangeIXT_EEERKNS0_2idIXT_EEE(%"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* sret(%"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item") align 8 %LocalItem.ascast, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable(16) %LocalSize.ascast, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable(16) %LocalId.ascast) #13
  call spir_func void @_ZN2cl4sycl6detail7Builder12createNDItemILi2EEENS0_7nd_itemIXT_EEERKNS0_4itemIXT_ELb1EEERKNS6_IXT_ELb0EEERKNS0_5groupIXT_EEE(%"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* sret(%"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item") align 8 %agg.result, %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* align 8 dereferenceable(48) %GlobalItem.ascast, %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* align 8 dereferenceable(32) %LocalItem.ascast, %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* align 8 dereferenceable(64) %Group.ascast) #13
  %11 = bitcast %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item"* %LocalItem to i8*
  call void @llvm.lifetime.end.p0i8(i64 32, i8* %11) #12
  %12 = bitcast %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item"* %GlobalItem to i8*
  call void @llvm.lifetime.end.p0i8(i64 48, i8* %12) #12
  %13 = bitcast %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group"* %Group to i8*
  call void @llvm.lifetime.end.p0i8(i64 64, i8* %13) #12
  %14 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %GlobalOffset to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %14) #12
  %15 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %LocalId to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %15) #12
  %16 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %GlobalId to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %16) #12
  %17 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %GroupId to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %17) #12
  %18 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %GroupRange to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %18) #12
  %19 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %LocalSize to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %19) #12
  %20 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %GlobalSize to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %20) #12
  ret void
}

; Function Attrs: convergent norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* @_ZN2cl4sycl6detail7declptrINS0_7nd_itemILi2EEEEEPT_v() #6 comdat {
entry:
  %retval = alloca %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)*, align 8
  %retval.ascast = addrspacecast %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)** %retval to %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* addrspace(4)*
  ret %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* null
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nofree nounwind willreturn writeonly
declare void @llvm.memset.p4i8.i64(i8 addrspace(4)* nocapture writeonly, i8, i64, i1 immarg) #7

; Function Attrs: convergent norecurse
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl2idILi2EEC2Ev(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable_or_null(16) %this) unnamed_addr #2 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %this, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %this1 to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  call spir_func void @_ZN2cl4sycl6detail5arrayILi2EEC2ILi2ELm0EEEv(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %0) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl6detail14InitializedValILi2ENS0_5rangeEE3getILi0EEENS3_ILi2EEEv(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.result) #3 comdat align 2 {
entry:
  call spir_func void @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable_or_null(16) %agg.result, i64 0, i64 0) #13
  ret void
}

; Function Attrs: convergent norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl6detail18AccessorImplDeviceILi2EEC2ENS0_2idILi2EEENS0_5rangeILi2EEES7_(%"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* align 8 dereferenceable_or_null(48) %this, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %Offset, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %AccessRange, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %MemoryRange) unnamed_addr #8 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %this, %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %Offset.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %Offset to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*
  %AccessRange.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %AccessRange to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %MemoryRange.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %MemoryRange to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  %this1 = load %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)*, %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %Offset2 = getelementptr inbounds %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice", %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %this1, i32 0, i32 0
  %0 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %Offset2 to i8 addrspace(4)*
  %1 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %Offset.ascast to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %0, i8 addrspace(4)* align 8 %1, i64 16, i1 false)
  %AccessRange3 = getelementptr inbounds %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice", %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %this1, i32 0, i32 1
  %2 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %AccessRange3 to i8 addrspace(4)*
  %3 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %AccessRange.ascast to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %2, i8 addrspace(4)* align 8 %3, i64 16, i1 false)
  %MemRange = getelementptr inbounds %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice", %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %this1, i32 0, i32 2
  %4 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %MemRange to i8 addrspace(4)*
  %5 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %MemoryRange.ascast to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %4, i8 addrspace(4)* align 8 %5, i64 16, i1 false)
  ret void
}

; Function Attrs: convergent norecurse
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl6detail5arrayILi2EEC2ILi2ELm0EEEv(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %this) unnamed_addr #2 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* %this, %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*, %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  call spir_func void @_ZN2cl4sycl6detail5arrayILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %this1, i64 0, i64 0) #13
  ret void
}

; Function Attrs: convergent norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl6detail5arrayILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %this, i64 %dim0, i64 %dim1) unnamed_addr #8 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* addrspace(4)*
  %dim0.addr = alloca i64, align 8
  %dim0.addr.ascast = addrspacecast i64* %dim0.addr to i64 addrspace(4)*
  %dim1.addr = alloca i64, align 8
  %dim1.addr.ascast = addrspacecast i64* %dim1.addr to i64 addrspace(4)*
  store %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* %this, %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  store i64 %dim0, i64 addrspace(4)* %dim0.addr.ascast, align 8, !tbaa !11
  store i64 %dim1, i64 addrspace(4)* %dim1.addr.ascast, align 8, !tbaa !11
  %this1 = load %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*, %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %common_array = getelementptr inbounds %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array", %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* %this1, i32 0, i32 0
  %arrayinit.begin = getelementptr inbounds [2 x i64], [2 x i64] addrspace(4)* %common_array, i64 0, i64 0
  %0 = load i64, i64 addrspace(4)* %dim0.addr.ascast, align 8, !tbaa !11
  store i64 %0, i64 addrspace(4)* %arrayinit.begin, align 8, !tbaa !11
  %arrayinit.element = getelementptr inbounds i64, i64 addrspace(4)* %arrayinit.begin, i64 1
  %1 = load i64, i64 addrspace(4)* %dim1.addr.ascast, align 8, !tbaa !11
  store i64 %1, i64 addrspace(4)* %arrayinit.element, align 8, !tbaa !11
  ret void
}

; Function Attrs: convergent norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable_or_null(16) %this, i64 %dim0, i64 %dim1) unnamed_addr #8 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)*
  %dim0.addr = alloca i64, align 8
  %dim0.addr.ascast = addrspacecast i64* %dim0.addr to i64 addrspace(4)*
  %dim1.addr = alloca i64, align 8
  %dim1.addr.ascast = addrspacecast i64* %dim1.addr to i64 addrspace(4)*
  store %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %this, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  store i64 %dim0, i64 addrspace(4)* %dim0.addr.ascast, align 8, !tbaa !11
  store i64 %dim1, i64 addrspace(4)* %dim1.addr.ascast, align 8, !tbaa !11
  %this1 = load %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %this1 to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %1 = load i64, i64 addrspace(4)* %dim0.addr.ascast, align 8, !tbaa !11
  %2 = load i64, i64 addrspace(4)* %dim1.addr.ascast, align 8, !tbaa !11
  call spir_func void @_ZN2cl4sycl6detail5arrayILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %0, i64 %1, i64 %2) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %this, i32 %dimension) #3 comdat align 2 {
entry:
  %this.addr.i = alloca %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*, align 8
  %dimension.addr.i = alloca i32, align 4
  %retval = alloca i64 addrspace(4)*, align 8
  %retval.ascast = addrspacecast i64 addrspace(4)** %retval to i64 addrspace(4)* addrspace(4)*
  %this.addr = alloca %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* addrspace(4)*
  %dimension.addr = alloca i32, align 4
  %dimension.addr.ascast = addrspacecast i32* %dimension.addr to i32 addrspace(4)*
  store %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* %this, %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  store i32 %dimension, i32 addrspace(4)* %dimension.addr.ascast, align 4, !tbaa !23
  %this1 = load %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*, %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = load i32, i32 addrspace(4)* %dimension.addr.ascast, align 4, !tbaa !23
  %this.addr.ascast.i = addrspacecast %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)** %this.addr.i to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* addrspace(4)*
  %dimension.addr.ascast.i = addrspacecast i32* %dimension.addr.i to i32 addrspace(4)*
  store %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* %this1, %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* addrspace(4)* %this.addr.ascast.i, align 8, !tbaa !7
  store i32 %0, i32 addrspace(4)* %dimension.addr.ascast.i, align 4, !tbaa !23
  %this1.i = load %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*, %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* addrspace(4)* %this.addr.ascast.i, align 8
  %common_array = getelementptr inbounds %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array", %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* %this1, i32 0, i32 0
  %1 = load i32, i32 addrspace(4)* %dimension.addr.ascast, align 4, !tbaa !23
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [2 x i64], [2 x i64] addrspace(4)* %common_array, i64 0, i64 %idxprom
  ret i64 addrspace(4)* %arrayidx
}

; Function Attrs: convergent norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* @_ZN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE9getOffsetEv(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this) #6 comdat align 2 {
entry:
  %retval = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, align 8
  %retval.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)** %retval to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)*
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %impl = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %Offset = getelementptr inbounds %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice", %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %impl, i32 0, i32 0
  ret %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %Offset
}

; Function Attrs: convergent norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* @_ZN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getAccessRangeEv(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this) #6 comdat align 2 {
entry:
  %retval = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, align 8
  %retval.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)** %retval to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)*
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %impl = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %AccessRange = getelementptr inbounds %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice", %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %impl, i32 0, i32 1
  ret %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %AccessRange
}

; Function Attrs: convergent norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* @_ZN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getMemoryRangeEv(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this) #6 comdat align 2 {
entry:
  %retval = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, align 8
  %retval.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)** %retval to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)*
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %impl = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %MemRange = getelementptr inbounds %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice", %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %impl, i32 0, i32 2
  ret %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %MemRange
}

; Function Attrs: convergent norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* @_ZN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE9getOffsetEv(%"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this) #6 comdat align 2 {
entry:
  %retval = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, align 8
  %retval.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)** %retval to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)*
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %impl = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %Offset = getelementptr inbounds %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice", %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %impl, i32 0, i32 0
  ret %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %Offset
}

; Function Attrs: convergent norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* @_ZN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getAccessRangeEv(%"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this) #6 comdat align 2 {
entry:
  %retval = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, align 8
  %retval.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)** %retval to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)*
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %impl = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %AccessRange = getelementptr inbounds %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice", %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %impl, i32 0, i32 1
  ret %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %AccessRange
}

; Function Attrs: convergent norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* @_ZN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getMemoryRangeEv(%"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this) #6 comdat align 2 {
entry:
  %retval = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, align 8
  %retval.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)** %retval to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)*
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %impl = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %MemRange = getelementptr inbounds %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice", %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %impl, i32 0, i32 2
  ret %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %MemRange
}

; Function Attrs: convergent norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func void @_ZNK2cl4sycl7nd_itemILi2EE13get_sub_groupEv(%"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)* noalias sret(%"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group") align 1 %agg.result, %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* align 8 dereferenceable_or_null(144) %this) #6 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* %this, %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)*, %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  ret void
}

; Function Attrs: convergent norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEC2ES7_(%"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* align 8 dereferenceable_or_null(8) %this, %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* byval(%"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group") align 1 %sg) unnamed_addr #8 comdat align 2 {
entry:
  %this.addr = alloca %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)** %this.addr to %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)*
  store %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %this, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %sg.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %sg to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)*
  %this1 = load %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  ret void
}

; Function Attrs: convergent norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEC2ES7_(%"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* align 8 dereferenceable_or_null(8) %this, %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* byval(%"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group") align 1 %sg) unnamed_addr #8 comdat align 2 {
entry:
  %this.addr = alloca %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)** %this.addr to %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)*
  store %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %this, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %sg.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %sg to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)*
  %this1 = load %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  ret void
}

; Function Attrs: convergent norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEC2ES7_(%"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* align 8 dereferenceable_or_null(8) %this, %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* byval(%"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group") align 1 %sg) unnamed_addr #8 comdat align 2 {
entry:
  %this.addr = alloca %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)** %this.addr to %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)*
  store %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* %this, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %sg.ascast = addrspacecast %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group"* %sg to %"struct._ZTSN2cl4sycl6ONEAPI9sub_groupE.cl::sycl::ONEAPI::sub_group" addrspace(4)*
  %this1 = load %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)*, %"struct._ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE.cl::sycl::ext::intel::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZNK2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE11get_pointerILS4_2014EvEENS0_9multi_ptrIfLNS2_13address_spaceE1EEEv(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr") align 8 %agg.result, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this) #3 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  %LinearIndex = alloca i64, align 8
  %LinearIndex.ascast = addrspacecast i64* %LinearIndex to i64 addrspace(4)*
  %agg.tmp = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id", align 8
  %agg.tmp.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %agg.tmp to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*
  %cleanup.dest.slot = alloca i32, align 4
  store %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = bitcast i64* %LinearIndex to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #12
  %1 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %agg.tmp.ascast to i8 addrspace(4)*
  call void @llvm.memset.p4i8.i64(i8 addrspace(4)* align 8 %1, i8 0, i64 16, i1 false)
  call spir_func void @_ZN2cl4sycl2idILi2EEC2Ev(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable_or_null(16) %agg.tmp.ascast) #13
  %agg.tmp.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %agg.tmp.ascast to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"*
  %call = call spir_func i64 @_ZNK2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getLinearIndexILi2EEEmNS0_2idIXT_EEE(%"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this1, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %agg.tmp.ascast.ascast) #13
  store i64 %call, i64 addrspace(4)* %LinearIndex.ascast, align 8, !tbaa !11
  %call2 = call spir_func float addrspace(1)* @_ZNK2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE15getQualifiedPtrEv(%"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this1) #13
  %2 = load i64, i64 addrspace(4)* %LinearIndex.ascast, align 8, !tbaa !11
  %add.ptr = getelementptr inbounds float, float addrspace(1)* %call2, i64 %2
  call spir_func void @_ZN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEC2EPU3AS1f(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %agg.result, float addrspace(1)* %add.ptr) #13
  %3 = bitcast i64* %LinearIndex to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %3) #12
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZNK2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEplEl(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr") align 8 %agg.result, %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %this, i64 %r) #3 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* addrspace(4)*
  %r.addr = alloca i64, align 8
  %r.addr.ascast = addrspacecast i64* %r.addr to i64 addrspace(4)*
  store %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* %this, %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  store i64 %r, i64 addrspace(4)* %r.addr.ascast, align 8, !tbaa !11
  %this1 = load %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*, %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %m_Pointer = getelementptr inbounds %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* %this1, i32 0, i32 0
  %0 = load float addrspace(1)*, float addrspace(1)* addrspace(4)* %m_Pointer, align 8, !tbaa !42
  %1 = load i64, i64 addrspace(4)* %r.addr.ascast, align 8, !tbaa !11
  %add.ptr = getelementptr inbounds float, float addrspace(1)* %0, i64 %1
  call spir_func void @_ZN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEC2EPU3AS1f(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %agg.result, float addrspace(1)* %add.ptr) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZNK2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE11get_pointerILS4_2014EvEENS0_9multi_ptrItLNS2_13address_spaceE1EEEv(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr") align 8 %agg.result, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this) #3 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  %LinearIndex = alloca i64, align 8
  %LinearIndex.ascast = addrspacecast i64* %LinearIndex to i64 addrspace(4)*
  %agg.tmp = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id", align 8
  %agg.tmp.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %agg.tmp to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*
  %cleanup.dest.slot = alloca i32, align 4
  store %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = bitcast i64* %LinearIndex to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #12
  %1 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %agg.tmp.ascast to i8 addrspace(4)*
  call void @llvm.memset.p4i8.i64(i8 addrspace(4)* align 8 %1, i8 0, i64 16, i1 false)
  call spir_func void @_ZN2cl4sycl2idILi2EEC2Ev(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable_or_null(16) %agg.tmp.ascast) #13
  %agg.tmp.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %agg.tmp.ascast to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"*
  %call = call spir_func i64 @_ZNK2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getLinearIndexILi2EEEmNS0_2idIXT_EEE(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this1, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %agg.tmp.ascast.ascast) #13
  store i64 %call, i64 addrspace(4)* %LinearIndex.ascast, align 8, !tbaa !11
  %call2 = call spir_func i16 addrspace(1)* @_ZNK2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE15getQualifiedPtrEv(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this1) #13
  %2 = load i64, i64 addrspace(4)* %LinearIndex.ascast, align 8, !tbaa !11
  %add.ptr = getelementptr inbounds i16, i16 addrspace(1)* %call2, i64 %2
  call spir_func void @_ZN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEC2EPU3AS1t(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %agg.result, i16 addrspace(1)* %add.ptr) #13
  %3 = bitcast i64* %LinearIndex to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %3) #12
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZNK2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEplEl(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr") align 8 %agg.result, %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %this, i64 %r) #3 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* addrspace(4)*
  %r.addr = alloca i64, align 8
  %r.addr.ascast = addrspacecast i64* %r.addr to i64 addrspace(4)*
  store %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* %this, %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  store i64 %r, i64 addrspace(4)* %r.addr.ascast, align 8, !tbaa !11
  %this1 = load %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*, %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %m_Pointer = getelementptr inbounds %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* %this1, i32 0, i32 0
  %0 = load i16 addrspace(1)*, i16 addrspace(1)* addrspace(4)* %m_Pointer, align 8, !tbaa !44
  %1 = load i64, i64 addrspace(4)* %r.addr.ascast, align 8, !tbaa !11
  %add.ptr = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 %1
  call spir_func void @_ZN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEC2EPU3AS1t(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %agg.result, i16 addrspace(1)* %add.ptr) #13
  ret void
}

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #9

; Function Attrs: convergent norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func i64 @_ZNK2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %this, i32 %dimension) #6 comdat align 2 {
entry:
  %this.addr.i = alloca %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*, align 8
  %dimension.addr.i = alloca i32, align 4
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %this.addr = alloca %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* addrspace(4)*
  %dimension.addr = alloca i32, align 4
  %dimension.addr.ascast = addrspacecast i32* %dimension.addr to i32 addrspace(4)*
  store %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* %this, %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  store i32 %dimension, i32 addrspace(4)* %dimension.addr.ascast, align 4, !tbaa !23
  %this1 = load %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*, %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = load i32, i32 addrspace(4)* %dimension.addr.ascast, align 4, !tbaa !23
  %this.addr.ascast.i = addrspacecast %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)** %this.addr.i to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* addrspace(4)*
  %dimension.addr.ascast.i = addrspacecast i32* %dimension.addr.i to i32 addrspace(4)*
  store %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* %this1, %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* addrspace(4)* %this.addr.ascast.i, align 8, !tbaa !7
  store i32 %0, i32 addrspace(4)* %dimension.addr.ascast.i, align 4, !tbaa !23
  %this1.i = load %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*, %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* addrspace(4)* %this.addr.ascast.i, align 8
  %common_array = getelementptr inbounds %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array", %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* %this1, i32 0, i32 0
  %1 = load i32, i32 addrspace(4)* %dimension.addr.ascast, align 4, !tbaa !23
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [2 x i64], [2 x i64] addrspace(4)* %common_array, i64 0, i64 %idxprom
  %2 = load i64, i64 addrspace(4)* %arrayidx, align 8, !tbaa !11
  ret i64 %2
}

; Function Attrs: convergent norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func float addrspace(1)* @_ZNK2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EE3getEv(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %this) #6 comdat align 2 {
entry:
  %retval = alloca float addrspace(1)*, align 8
  %retval.ascast = addrspacecast float addrspace(1)** %retval to float addrspace(1)* addrspace(4)*
  %this.addr = alloca %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* %this, %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*, %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %m_Pointer = getelementptr inbounds %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* %this1, i32 0, i32 0
  %0 = load float addrspace(1)*, float addrspace(1)* addrspace(4)* %m_Pointer, align 8, !tbaa !42
  ret float addrspace(1)* %0
}

; Function Attrs: convergent
declare dso_local spir_func %"struct._ZTSN5__spv19__spirv_MatrixINTELIfLm47ELm47ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* @_Z23__spirv_MatrixLoadINTELIfLm47ELm47ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEPNS0_19__spirv_MatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPS5_mS1_S3_i(float addrspace(4)*, i64, i32, i32, i32) #10

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func i64 @_ZNK2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getLinearIndexILi2EEEmNS0_2idIXT_EEE(%"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %Id) #3 comdat align 2 {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  %Result = alloca i64, align 8
  %Result.ascast = addrspacecast i64* %Result to i64 addrspace(4)*
  %I = alloca i32, align 4
  %I.ascast = addrspacecast i32* %I to i32 addrspace(4)*
  %cleanup.dest.slot = alloca i32, align 4
  %cleanup.dest.slot7 = alloca i32, align 4
  store %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %Id.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %Id to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*
  %this1 = load %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = bitcast i64* %Result to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #12
  store i64 0, i64 addrspace(4)* %Result.ascast, align 8, !tbaa !11
  %1 = bitcast i32* %I to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %1) #12
  store i32 0, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %2 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %cmp = icmp slt i32 %2, 2
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  %3 = bitcast i32* %I to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %3) #12
  br label %for.end

for.body:                                         ; preds = %for.cond
  %4 = load i64, i64 addrspace(4)* %Result.ascast, align 8, !tbaa !11
  %call = call spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* @_ZNK2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getMemoryRangeEv(%"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this1) #13
  %5 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %call to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %6 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %call2 = call spir_func i64 @_ZNK2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %5, i32 %6) #13
  %mul = mul i64 %4, %call2
  %call3 = call spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* @_ZNK2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE9getOffsetEv(%"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this1) #13
  %7 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %call3 to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %8 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %call4 = call spir_func i64 @_ZNK2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %7, i32 %8) #13
  %add = add i64 %mul, %call4
  %9 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %Id.ascast to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %10 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %call5 = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %9, i32 %10) #13
  %11 = load i64, i64 addrspace(4)* %call5, align 8, !tbaa !11
  %add6 = add i64 %add, %11
  store i64 %add6, i64 addrspace(4)* %Result.ascast, align 8, !tbaa !11
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %12 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %inc = add nsw i32 %12, 1
  store i32 %inc, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  br label %for.cond, !llvm.loop !46

for.end:                                          ; preds = %for.cond.cleanup
  %13 = load i64, i64 addrspace(4)* %Result.ascast, align 8, !tbaa !11
  %14 = bitcast i64* %Result to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %14) #12
  ret i64 %13
}

; Function Attrs: convergent norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func float addrspace(1)* @_ZNK2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE15getQualifiedPtrEv(%"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this) #6 comdat align 2 {
entry:
  %retval = alloca float addrspace(1)*, align 8
  %retval.ascast = addrspacecast float addrspace(1)** %retval to float addrspace(1)* addrspace(4)*
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 1
  %MData = bitcast %union._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEUt_E.anon addrspace(4)* %0 to float addrspace(1)* addrspace(4)*
  %1 = load float addrspace(1)*, float addrspace(1)* addrspace(4)* %MData, align 8, !tbaa !22
  ret float addrspace(1)* %1
}

; Function Attrs: convergent norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEC2EPU3AS1f(%"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %this, float addrspace(1)* %pointer) unnamed_addr #8 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* addrspace(4)*
  %pointer.addr = alloca float addrspace(1)*, align 8
  %pointer.addr.ascast = addrspacecast float addrspace(1)** %pointer.addr to float addrspace(1)* addrspace(4)*
  store %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* %this, %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  store float addrspace(1)* %pointer, float addrspace(1)* addrspace(4)* %pointer.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*, %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %m_Pointer = getelementptr inbounds %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", %"class._ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* %this1, i32 0, i32 0
  %0 = load float addrspace(1)*, float addrspace(1)* addrspace(4)* %pointer.addr.ascast, align 8, !tbaa !7
  store float addrspace(1)* %0, float addrspace(1)* addrspace(4)* %m_Pointer, align 8, !tbaa !42
  ret void
}

; Function Attrs: convergent norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* @_ZNK2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getMemoryRangeEv(%"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this) #6 comdat align 2 {
entry:
  %retval = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, align 8
  %retval.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)** %retval to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)*
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %impl = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %MemRange = getelementptr inbounds %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice", %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %impl, i32 0, i32 2
  ret %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %MemRange
}

; Function Attrs: convergent norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* @_ZNK2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE9getOffsetEv(%"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this) #6 comdat align 2 {
entry:
  %retval = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, align 8
  %retval.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)** %retval to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)*
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %impl = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %Offset = getelementptr inbounds %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice", %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %impl, i32 0, i32 0
  ret %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %Offset
}

; Function Attrs: convergent norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func i16 addrspace(1)* @_ZNK2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EE3getEv(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %this) #6 comdat align 2 {
entry:
  %retval = alloca i16 addrspace(1)*, align 8
  %retval.ascast = addrspacecast i16 addrspace(1)** %retval to i16 addrspace(1)* addrspace(4)*
  %this.addr = alloca %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* %this, %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*, %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %m_Pointer = getelementptr inbounds %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* %this1, i32 0, i32 0
  %0 = load i16 addrspace(1)*, i16 addrspace(1)* addrspace(4)* %m_Pointer, align 8, !tbaa !44
  ret i16 addrspace(1)* %0
}

; Function Attrs: convergent
declare dso_local spir_func %"struct._ZTSN5__spv19__spirv_MatrixINTELItLm47ELm146ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* @_Z23__spirv_MatrixLoadINTELItLm47ELm146ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEPNS0_19__spirv_MatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPS5_mS1_S3_i(i16 addrspace(4)*, i64, i32, i32, i32) #10

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func i64 @_ZNK2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getLinearIndexILi2EEEmNS0_2idIXT_EEE(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %Id) #3 comdat align 2 {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  %Result = alloca i64, align 8
  %Result.ascast = addrspacecast i64* %Result to i64 addrspace(4)*
  %I = alloca i32, align 4
  %I.ascast = addrspacecast i32* %I to i32 addrspace(4)*
  %cleanup.dest.slot = alloca i32, align 4
  %cleanup.dest.slot7 = alloca i32, align 4
  store %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %Id.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %Id to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*
  %this1 = load %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = bitcast i64* %Result to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #12
  store i64 0, i64 addrspace(4)* %Result.ascast, align 8, !tbaa !11
  %1 = bitcast i32* %I to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %1) #12
  store i32 0, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %2 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %cmp = icmp slt i32 %2, 2
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  %3 = bitcast i32* %I to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %3) #12
  br label %for.end

for.body:                                         ; preds = %for.cond
  %4 = load i64, i64 addrspace(4)* %Result.ascast, align 8, !tbaa !11
  %call = call spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* @_ZNK2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getMemoryRangeEv(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this1) #13
  %5 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %call to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %6 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %call2 = call spir_func i64 @_ZNK2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %5, i32 %6) #13
  %mul = mul i64 %4, %call2
  %call3 = call spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* @_ZNK2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE9getOffsetEv(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this1) #13
  %7 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %call3 to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %8 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %call4 = call spir_func i64 @_ZNK2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %7, i32 %8) #13
  %add = add i64 %mul, %call4
  %9 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %Id.ascast to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %10 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %call5 = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi2EEixEi(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %9, i32 %10) #13
  %11 = load i64, i64 addrspace(4)* %call5, align 8, !tbaa !11
  %add6 = add i64 %add, %11
  store i64 %add6, i64 addrspace(4)* %Result.ascast, align 8, !tbaa !11
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %12 = load i32, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  %inc = add nsw i32 %12, 1
  store i32 %inc, i32 addrspace(4)* %I.ascast, align 4, !tbaa !23
  br label %for.cond, !llvm.loop !47

for.end:                                          ; preds = %for.cond.cleanup
  %13 = load i64, i64 addrspace(4)* %Result.ascast, align 8, !tbaa !11
  %14 = bitcast i64* %Result to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %14) #12
  ret i64 %13
}

; Function Attrs: convergent norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func i16 addrspace(1)* @_ZNK2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE15getQualifiedPtrEv(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this) #6 comdat align 2 {
entry:
  %retval = alloca i16 addrspace(1)*, align 8
  %retval.ascast = addrspacecast i16 addrspace(1)** %retval to i16 addrspace(1)* addrspace(4)*
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 1
  %MData = bitcast %union._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEUt_E.anon addrspace(4)* %0 to i16 addrspace(1)* addrspace(4)*
  %1 = load i16 addrspace(1)*, i16 addrspace(1)* addrspace(4)* %MData, align 8, !tbaa !22
  ret i16 addrspace(1)* %1
}

; Function Attrs: convergent norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEC2EPU3AS1t(%"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* align 8 dereferenceable_or_null(8) %this, i16 addrspace(1)* %pointer) unnamed_addr #8 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* addrspace(4)*
  %pointer.addr = alloca i16 addrspace(1)*, align 8
  %pointer.addr.ascast = addrspacecast i16 addrspace(1)** %pointer.addr to i16 addrspace(1)* addrspace(4)*
  store %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* %this, %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  store i16 addrspace(1)* %pointer, i16 addrspace(1)* addrspace(4)* %pointer.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)*, %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %m_Pointer = getelementptr inbounds %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr", %"class._ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE.cl::sycl::multi_ptr" addrspace(4)* %this1, i32 0, i32 0
  %0 = load i16 addrspace(1)*, i16 addrspace(1)* addrspace(4)* %pointer.addr.ascast, align 8, !tbaa !7
  store i16 addrspace(1)* %0, i16 addrspace(1)* addrspace(4)* %m_Pointer, align 8, !tbaa !44
  ret void
}

; Function Attrs: convergent norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* @_ZNK2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE14getMemoryRangeEv(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this) #6 comdat align 2 {
entry:
  %retval = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, align 8
  %retval.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)** %retval to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)*
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %impl = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %MemRange = getelementptr inbounds %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice", %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %impl, i32 0, i32 2
  ret %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %MemRange
}

; Function Attrs: convergent norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func align 8 dereferenceable(16) %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* @_ZNK2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEE9getOffsetEv(%"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(56) %this) #6 comdat align 2 {
entry:
  %retval = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, align 8
  %retval.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)** %retval to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)*
  %this.addr = alloca %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)*, %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %impl = getelementptr inbounds %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor", %"class._ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %Offset = getelementptr inbounds %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice", %"class._ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %impl, i32 0, i32 0
  ret %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %Offset
}

; Function Attrs: convergent
declare dso_local spir_func %"struct._ZTSN5__spv19__spirv_MatrixINTELItLm146ELm47ELNS_12MatrixLayoutE3ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* @_Z23__spirv_MatrixLoadINTELItLm146ELm47ELN5__spv12MatrixLayoutE3ELNS0_5Scope4FlagE3EEPNS0_19__spirv_MatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPS5_mS1_S3_i(i16 addrspace(4)*, i64, i32, i32, i32) #10

; Function Attrs: convergent
declare dso_local spir_func %"struct._ZTSN5__spv19__spirv_MatrixINTELIfLm47ELm47ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)* @_Z22__spirv_MatrixMadINTELItfLm47ELm146ELm47ELN5__spv12MatrixLayoutE0ELS1_3ELS1_0ELNS0_5Scope4FlagE3EEPNS0_19__spirv_MatrixINTELIT0_XT1_EXT3_EXT6_EXT7_EEEPNS4_IT_XT1_EXT2_EXT4_EXT7_EEEPNS4_IS8_XT2_EXT3_EXT5_EXT7_EEES7_S3_(%"struct._ZTSN5__spv19__spirv_MatrixINTELItLm47ELm146ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)*, %"struct._ZTSN5__spv19__spirv_MatrixINTELItLm146ELm47ELNS_12MatrixLayoutE3ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)*, %"struct._ZTSN5__spv19__spirv_MatrixINTELIfLm47ELm47ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)*, i32) #10

; Function Attrs: convergent
declare dso_local spir_func void @_Z24__spirv_MatrixStoreINTELIfLm47ELm47ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEvPT_PNS0_19__spirv_MatrixINTELIS4_XT0_EXT1_EXT2_EXT3_EEEmS1_S3_i(float addrspace(4)*, %"struct._ZTSN5__spv19__spirv_MatrixINTELIfLm47ELm47ELNS_12MatrixLayoutE0ELNS_5Scope4FlagE3EEE.__spv::__spirv_MatrixINTEL" addrspace(4)*, i64, i32, i32, i32) #10

; Function Attrs: convergent norecurse mustprogress
define internal spir_func void @_ZN7__spirvL14initGlobalSizeILi2EN2cl4sycl5rangeILi2EEEEET0_v(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.result) #3 {
entry:
  call spir_func void @_ZN7__spirv21InitSizesSTGlobalSizeILi2EN2cl4sycl5rangeILi2EEEE8initSizeEv(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* sret(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.result) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func void @_ZN7__spirvL17initWorkgroupSizeILi2EN2cl4sycl5rangeILi2EEEEET0_v(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.result) #3 {
entry:
  call spir_func void @_ZN7__spirv24InitSizesSTWorkgroupSizeILi2EN2cl4sycl5rangeILi2EEEE8initSizeEv(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* sret(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.result) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func void @_ZN7__spirvL17initNumWorkgroupsILi2EN2cl4sycl5rangeILi2EEEEET0_v(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.result) #3 {
entry:
  call spir_func void @_ZN7__spirv24InitSizesSTNumWorkgroupsILi2EN2cl4sycl5rangeILi2EEEE8initSizeEv(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* sret(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.result) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func void @_ZN7__spirvL15initWorkgroupIdILi2EN2cl4sycl2idILi2EEEEET0_v(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %agg.result) #3 {
entry:
  call spir_func void @_ZN7__spirv22InitSizesSTWorkgroupIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* sret(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %agg.result) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func void @_ZN7__spirvL22initGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEEET0_v(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %agg.result) #3 {
entry:
  call spir_func void @_ZN7__spirv29InitSizesSTGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* sret(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %agg.result) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func void @_ZN7__spirvL21initLocalInvocationIdILi2EN2cl4sycl2idILi2EEEEET0_v(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %agg.result) #3 {
entry:
  call spir_func void @_ZN7__spirv28InitSizesSTLocalInvocationIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* sret(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %agg.result) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func void @_ZN7__spirvL16initGlobalOffsetILi2EN2cl4sycl2idILi2EEEEET0_v(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %agg.result) #3 {
entry:
  call spir_func void @_ZN7__spirv23InitSizesSTGlobalOffsetILi2EN2cl4sycl2idILi2EEEE8initSizeEv(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* sret(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %agg.result) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl6detail7Builder11createGroupILi2EEENS0_5groupIXT_EEERKNS0_5rangeIXT_EEES9_S9_RKNS0_2idIXT_EEE(%"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group") align 8 %agg.result, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable(16) %Global, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable(16) %Local, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable(16) %Group, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable(16) %Index) #3 comdat align 2 {
entry:
  %Global.addr = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, align 8
  %Global.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)** %Global.addr to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)*
  %Local.addr = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, align 8
  %Local.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)** %Local.addr to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)*
  %Group.addr = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, align 8
  %Group.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)** %Group.addr to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)*
  %Index.addr = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, align 8
  %Index.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)** %Index.addr to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)*
  %agg.tmp = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", align 8
  %agg.tmp.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %agg.tmp to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  store %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %Global, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %Global.addr.ascast, align 8, !tbaa !7
  store %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %Local, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %Local.addr.ascast, align 8, !tbaa !7
  store %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %Group, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %Group.addr.ascast, align 8, !tbaa !7
  store %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %Index, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %Index.addr.ascast, align 8, !tbaa !7
  %0 = load %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %Global.addr.ascast, align 8, !tbaa !7
  %1 = load %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %Local.addr.ascast, align 8, !tbaa !7
  %2 = load %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %Group.addr.ascast, align 8, !tbaa !7
  %3 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %agg.tmp.ascast to i8 addrspace(4)*
  %4 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %2 to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %3, i8 addrspace(4)* align 8 %4, i64 16, i1 false)
  %5 = load %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %Index.addr.ascast, align 8, !tbaa !7
  %agg.tmp.ascast.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %agg.tmp.ascast to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"*
  call spir_func void @_ZN2cl4sycl5groupILi2EEC2ERKNS0_5rangeILi2EEES6_S4_RKNS0_2idILi2EEE(%"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* align 8 dereferenceable_or_null(64) %agg.result, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable(16) %0, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable(16) %1, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.tmp.ascast.ascast, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable(16) %5) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl6detail7Builder10createItemILi2ELb1EEENSt9enable_ifIXT0_ENS0_4itemIXT_EXT0_EEEE4typeERKNS0_5rangeIXT_EEERKNS0_2idIXT_EEESG_(%"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item") align 8 %agg.result, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable(16) %Extent, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable(16) %Index, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable(16) %Offset) #3 comdat align 2 {
entry:
  %Extent.addr = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, align 8
  %Extent.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)** %Extent.addr to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)*
  %Index.addr = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, align 8
  %Index.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)** %Index.addr to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)*
  %Offset.addr = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, align 8
  %Offset.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)** %Offset.addr to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %Extent, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %Extent.addr.ascast, align 8, !tbaa !7
  store %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %Index, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %Index.addr.ascast, align 8, !tbaa !7
  store %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %Offset, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %Offset.addr.ascast, align 8, !tbaa !7
  %0 = load %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %Extent.addr.ascast, align 8, !tbaa !7
  %1 = load %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %Index.addr.ascast, align 8, !tbaa !7
  %2 = load %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %Offset.addr.ascast, align 8, !tbaa !7
  call spir_func void @_ZN2cl4sycl4itemILi2ELb1EEC2ILb1EEERNSt9enable_ifIXT_EKNS0_5rangeILi2EEEE4typeERKNS0_2idILi2EEESE_(%"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* align 8 dereferenceable_or_null(48) %agg.result, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable(16) %0, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable(16) %1, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable(16) %2) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl6detail7Builder10createItemILi2ELb0EEENSt9enable_ifIXntT0_ENS0_4itemIXT_EXT0_EEEE4typeERKNS0_5rangeIXT_EEERKNS0_2idIXT_EEE(%"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item") align 8 %agg.result, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable(16) %Extent, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable(16) %Index) #3 comdat align 2 {
entry:
  %Extent.addr = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, align 8
  %Extent.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)** %Extent.addr to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)*
  %Index.addr = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, align 8
  %Index.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)** %Index.addr to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %Extent, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %Extent.addr.ascast, align 8, !tbaa !7
  store %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %Index, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %Index.addr.ascast, align 8, !tbaa !7
  %0 = load %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %Extent.addr.ascast, align 8, !tbaa !7
  %1 = load %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %Index.addr.ascast, align 8, !tbaa !7
  call spir_func void @_ZN2cl4sycl4itemILi2ELb0EEC2ILb0EEERNSt9enable_ifIXntT_EKNS0_5rangeILi2EEEE4typeERKNS0_2idILi2EEE(%"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* align 8 dereferenceable_or_null(32) %agg.result, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable(16) %0, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable(16) %1) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl6detail7Builder12createNDItemILi2EEENS0_7nd_itemIXT_EEERKNS0_4itemIXT_ELb1EEERKNS6_IXT_ELb0EEERKNS0_5groupIXT_EEE(%"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item") align 8 %agg.result, %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* align 8 dereferenceable(48) %Global, %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* align 8 dereferenceable(32) %Local, %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* align 8 dereferenceable(64) %Group) #3 comdat align 2 {
entry:
  %Global.addr = alloca %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)*, align 8
  %Global.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)** %Global.addr to %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* addrspace(4)*
  %Local.addr = alloca %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)*, align 8
  %Local.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)** %Local.addr to %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* addrspace(4)*
  %Group.addr = alloca %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)*, align 8
  %Group.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)** %Group.addr to %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* %Global, %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* addrspace(4)* %Global.addr.ascast, align 8, !tbaa !7
  store %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* %Local, %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* addrspace(4)* %Local.addr.ascast, align 8, !tbaa !7
  store %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* %Group, %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* addrspace(4)* %Group.addr.ascast, align 8, !tbaa !7
  %0 = load %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)*, %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* addrspace(4)* %Global.addr.ascast, align 8, !tbaa !7
  %1 = load %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)*, %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* addrspace(4)* %Local.addr.ascast, align 8, !tbaa !7
  %2 = load %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)*, %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* addrspace(4)* %Group.addr.ascast, align 8, !tbaa !7
  call spir_func void @_ZN2cl4sycl7nd_itemILi2EEC2ERKNS0_4itemILi2ELb1EEERKNS3_ILi2ELb0EEERKNS0_5groupILi2EEE(%"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* align 8 dereferenceable_or_null(144) %agg.result, %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* align 8 dereferenceable(48) %0, %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* align 8 dereferenceable(32) %1, %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* align 8 dereferenceable(64) %2) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZN7__spirv21InitSizesSTGlobalSizeILi2EN2cl4sycl5rangeILi2EEEE8initSizeEv(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.result) #3 comdat align 2 {
entry:
  %call = call spir_func i64 @_ZN7__spirvL13getGlobalSizeILi1EEEmv() #13
  %call1 = call spir_func i64 @_ZN7__spirvL13getGlobalSizeILi0EEEmv() #13
  call spir_func void @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable_or_null(16) %agg.result, i64 %call, i64 %call1) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func i64 @_ZN7__spirvL13getGlobalSizeILi1EEEmv() #3 {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %call = call spir_func i64 @_Z20__spirv_GlobalSize_yv() #13
  ret i64 %call
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func i64 @_ZN7__spirvL13getGlobalSizeILi0EEEmv() #3 {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %call = call spir_func i64 @_Z20__spirv_GlobalSize_xv() #13
  ret i64 %call
}

; Function Attrs: convergent inlinehint norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func i64 @_Z20__spirv_GlobalSize_yv() #11 comdat {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalSize to <3 x i64> addrspace(4)*), align 32
  %1 = extractelement <3 x i64> %0, i64 1
  ret i64 %1
}

; Function Attrs: convergent inlinehint norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func i64 @_Z20__spirv_GlobalSize_xv() #11 comdat {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalSize to <3 x i64> addrspace(4)*), align 32
  %1 = extractelement <3 x i64> %0, i64 0
  ret i64 %1
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZN7__spirv24InitSizesSTWorkgroupSizeILi2EN2cl4sycl5rangeILi2EEEE8initSizeEv(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.result) #3 comdat align 2 {
entry:
  %call = call spir_func i64 @_ZN7__spirvL16getWorkgroupSizeILi1EEEmv() #13
  %call1 = call spir_func i64 @_ZN7__spirvL16getWorkgroupSizeILi0EEEmv() #13
  call spir_func void @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable_or_null(16) %agg.result, i64 %call, i64 %call1) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func i64 @_ZN7__spirvL16getWorkgroupSizeILi1EEEmv() #3 {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %call = call spir_func i64 @_Z23__spirv_WorkgroupSize_yv() #13
  ret i64 %call
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func i64 @_ZN7__spirvL16getWorkgroupSizeILi0EEEmv() #3 {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %call = call spir_func i64 @_Z23__spirv_WorkgroupSize_xv() #13
  ret i64 %call
}

; Function Attrs: convergent inlinehint norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func i64 @_Z23__spirv_WorkgroupSize_yv() #11 comdat {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupSize to <3 x i64> addrspace(4)*), align 32
  %1 = extractelement <3 x i64> %0, i64 1
  ret i64 %1
}

; Function Attrs: convergent inlinehint norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func i64 @_Z23__spirv_WorkgroupSize_xv() #11 comdat {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupSize to <3 x i64> addrspace(4)*), align 32
  %1 = extractelement <3 x i64> %0, i64 0
  ret i64 %1
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZN7__spirv24InitSizesSTNumWorkgroupsILi2EN2cl4sycl5rangeILi2EEEE8initSizeEv(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %agg.result) #3 comdat align 2 {
entry:
  %call = call spir_func i64 @_ZN7__spirvL16getNumWorkgroupsILi1EEEmv() #13
  %call1 = call spir_func i64 @_ZN7__spirvL16getNumWorkgroupsILi0EEEmv() #13
  call spir_func void @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable_or_null(16) %agg.result, i64 %call, i64 %call1) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func i64 @_ZN7__spirvL16getNumWorkgroupsILi1EEEmv() #3 {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %call = call spir_func i64 @_Z23__spirv_NumWorkgroups_yv() #13
  ret i64 %call
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func i64 @_ZN7__spirvL16getNumWorkgroupsILi0EEEmv() #3 {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %call = call spir_func i64 @_Z23__spirv_NumWorkgroups_xv() #13
  ret i64 %call
}

; Function Attrs: convergent inlinehint norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func i64 @_Z23__spirv_NumWorkgroups_yv() #11 comdat {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInNumWorkgroups to <3 x i64> addrspace(4)*), align 32
  %1 = extractelement <3 x i64> %0, i64 1
  ret i64 %1
}

; Function Attrs: convergent inlinehint norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func i64 @_Z23__spirv_NumWorkgroups_xv() #11 comdat {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInNumWorkgroups to <3 x i64> addrspace(4)*), align 32
  %1 = extractelement <3 x i64> %0, i64 0
  ret i64 %1
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZN7__spirv22InitSizesSTWorkgroupIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %agg.result) #3 comdat align 2 {
entry:
  %call = call spir_func i64 @_ZN7__spirvL14getWorkgroupIdILi1EEEmv() #13
  %call1 = call spir_func i64 @_ZN7__spirvL14getWorkgroupIdILi0EEEmv() #13
  call spir_func void @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable_or_null(16) %agg.result, i64 %call, i64 %call1) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func i64 @_ZN7__spirvL14getWorkgroupIdILi1EEEmv() #3 {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %call = call spir_func i64 @_Z21__spirv_WorkgroupId_yv() #13
  ret i64 %call
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func i64 @_ZN7__spirvL14getWorkgroupIdILi0EEEmv() #3 {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %call = call spir_func i64 @_Z21__spirv_WorkgroupId_xv() #13
  ret i64 %call
}

; Function Attrs: convergent norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable_or_null(16) %this, i64 %dim0, i64 %dim1) unnamed_addr #8 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)*
  %dim0.addr = alloca i64, align 8
  %dim0.addr.ascast = addrspacecast i64* %dim0.addr to i64 addrspace(4)*
  %dim1.addr = alloca i64, align 8
  %dim1.addr.ascast = addrspacecast i64* %dim1.addr to i64 addrspace(4)*
  store %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %this, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  store i64 %dim0, i64 addrspace(4)* %dim0.addr.ascast, align 8, !tbaa !11
  store i64 %dim1, i64 addrspace(4)* %dim1.addr.ascast, align 8, !tbaa !11
  %this1 = load %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %this1 to %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)*
  %1 = load i64, i64 addrspace(4)* %dim0.addr.ascast, align 8, !tbaa !11
  %2 = load i64, i64 addrspace(4)* %dim1.addr.ascast, align 8, !tbaa !11
  call spir_func void @_ZN2cl4sycl6detail5arrayILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm(%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(16) %0, i64 %1, i64 %2) #13
  ret void
}

; Function Attrs: convergent inlinehint norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func i64 @_Z21__spirv_WorkgroupId_yv() #11 comdat {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupId to <3 x i64> addrspace(4)*), align 32
  %1 = extractelement <3 x i64> %0, i64 1
  ret i64 %1
}

; Function Attrs: convergent inlinehint norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func i64 @_Z21__spirv_WorkgroupId_xv() #11 comdat {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupId to <3 x i64> addrspace(4)*), align 32
  %1 = extractelement <3 x i64> %0, i64 0
  ret i64 %1
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZN7__spirv29InitSizesSTGlobalInvocationIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %agg.result) #3 comdat align 2 {
entry:
  %call = call spir_func i64 @_ZN7__spirvL21getGlobalInvocationIdILi1EEEmv() #13
  %call1 = call spir_func i64 @_ZN7__spirvL21getGlobalInvocationIdILi0EEEmv() #13
  call spir_func void @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable_or_null(16) %agg.result, i64 %call, i64 %call1) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func i64 @_ZN7__spirvL21getGlobalInvocationIdILi1EEEmv() #3 {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %call = call spir_func i64 @_Z28__spirv_GlobalInvocationId_yv() #13
  ret i64 %call
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func i64 @_ZN7__spirvL21getGlobalInvocationIdILi0EEEmv() #3 {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %call = call spir_func i64 @_Z28__spirv_GlobalInvocationId_xv() #13
  ret i64 %call
}

; Function Attrs: convergent inlinehint norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_yv() #11 comdat {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId to <3 x i64> addrspace(4)*), align 32
  %1 = extractelement <3 x i64> %0, i64 1
  ret i64 %1
}

; Function Attrs: convergent inlinehint norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func i64 @_Z28__spirv_GlobalInvocationId_xv() #11 comdat {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId to <3 x i64> addrspace(4)*), align 32
  %1 = extractelement <3 x i64> %0, i64 0
  ret i64 %1
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZN7__spirv28InitSizesSTLocalInvocationIdILi2EN2cl4sycl2idILi2EEEE8initSizeEv(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %agg.result) #3 comdat align 2 {
entry:
  %call = call spir_func i64 @_ZN7__spirvL20getLocalInvocationIdILi1EEEmv() #13
  %call1 = call spir_func i64 @_ZN7__spirvL20getLocalInvocationIdILi0EEEmv() #13
  call spir_func void @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable_or_null(16) %agg.result, i64 %call, i64 %call1) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func i64 @_ZN7__spirvL20getLocalInvocationIdILi1EEEmv() #3 {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %call = call spir_func i64 @_Z27__spirv_LocalInvocationId_yv() #13
  ret i64 %call
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func i64 @_ZN7__spirvL20getLocalInvocationIdILi0EEEmv() #3 {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %call = call spir_func i64 @_Z27__spirv_LocalInvocationId_xv() #13
  ret i64 %call
}

; Function Attrs: convergent inlinehint norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func i64 @_Z27__spirv_LocalInvocationId_yv() #11 comdat {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInLocalInvocationId to <3 x i64> addrspace(4)*), align 32
  %1 = extractelement <3 x i64> %0, i64 1
  ret i64 %1
}

; Function Attrs: convergent inlinehint norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func i64 @_Z27__spirv_LocalInvocationId_xv() #11 comdat {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInLocalInvocationId to <3 x i64> addrspace(4)*), align 32
  %1 = extractelement <3 x i64> %0, i64 0
  ret i64 %1
}

; Function Attrs: convergent norecurse mustprogress
define linkonce_odr dso_local spir_func void @_ZN7__spirv23InitSizesSTGlobalOffsetILi2EN2cl4sycl2idILi2EEEE8initSizeEv(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* noalias sret(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %agg.result) #3 comdat align 2 {
entry:
  %call = call spir_func i64 @_ZN7__spirvL15getGlobalOffsetILi1EEEmv() #13
  %call1 = call spir_func i64 @_ZN7__spirvL15getGlobalOffsetILi0EEEmv() #13
  call spir_func void @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable_or_null(16) %agg.result, i64 %call, i64 %call1) #13
  ret void
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func i64 @_ZN7__spirvL15getGlobalOffsetILi1EEEmv() #3 {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %call = call spir_func i64 @_Z22__spirv_GlobalOffset_yv() #13
  ret i64 %call
}

; Function Attrs: convergent norecurse mustprogress
define internal spir_func i64 @_ZN7__spirvL15getGlobalOffsetILi0EEEmv() #3 {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %call = call spir_func i64 @_Z22__spirv_GlobalOffset_xv() #13
  ret i64 %call
}

; Function Attrs: convergent inlinehint norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func i64 @_Z22__spirv_GlobalOffset_yv() #11 comdat {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalOffset to <3 x i64> addrspace(4)*), align 32
  %1 = extractelement <3 x i64> %0, i64 1
  ret i64 %1
}

; Function Attrs: convergent inlinehint norecurse nounwind mustprogress
define linkonce_odr dso_local spir_func i64 @_Z22__spirv_GlobalOffset_xv() #11 comdat {
entry:
  %retval = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalOffset to <3 x i64> addrspace(4)*), align 32
  %1 = extractelement <3 x i64> %0, i64 0
  ret i64 %1
}

; Function Attrs: convergent norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl5groupILi2EEC2ERKNS0_5rangeILi2EEES6_S4_RKNS0_2idILi2EEE(%"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* align 8 dereferenceable_or_null(64) %this, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable(16) %G, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable(16) %L, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %GroupRange, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable(16) %I) unnamed_addr #8 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* addrspace(4)*
  %G.addr = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, align 8
  %G.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)** %G.addr to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)*
  %L.addr = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, align 8
  %L.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)** %L.addr to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)*
  %I.addr = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, align 8
  %I.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)** %I.addr to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* %this, %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  store %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %G, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %G.addr.ascast, align 8, !tbaa !7
  store %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %L, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %L.addr.ascast, align 8, !tbaa !7
  %GroupRange.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %GroupRange to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*
  store %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %I, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %I.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)*, %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %globalRange = getelementptr inbounds %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group", %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* %this1, i32 0, i32 0
  %0 = load %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %G.addr.ascast, align 8, !tbaa !7
  %1 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %globalRange to i8 addrspace(4)*
  %2 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %0 to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %1, i8 addrspace(4)* align 8 %2, i64 16, i1 false)
  %localRange = getelementptr inbounds %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group", %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* %this1, i32 0, i32 1
  %3 = load %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %L.addr.ascast, align 8, !tbaa !7
  %4 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %localRange to i8 addrspace(4)*
  %5 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %3 to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %4, i8 addrspace(4)* align 8 %5, i64 16, i1 false)
  %groupRange = getelementptr inbounds %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group", %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* %this1, i32 0, i32 2
  %6 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %groupRange to i8 addrspace(4)*
  %7 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %GroupRange.ascast to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %6, i8 addrspace(4)* align 8 %7, i64 16, i1 false)
  %index = getelementptr inbounds %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group", %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* %this1, i32 0, i32 3
  %8 = load %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %I.addr.ascast, align 8, !tbaa !7
  %9 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %index to i8 addrspace(4)*
  %10 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %8 to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %9, i8 addrspace(4)* align 8 %10, i64 16, i1 false)
  ret void
}

; Function Attrs: convergent norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl4itemILi2ELb1EEC2ILb1EEERNSt9enable_ifIXT_EKNS0_5rangeILi2EEEE4typeERKNS0_2idILi2EEESE_(%"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* align 8 dereferenceable_or_null(48) %this, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable(16) %extent, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable(16) %index, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable(16) %offset) unnamed_addr #8 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* addrspace(4)*
  %extent.addr = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, align 8
  %extent.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)** %extent.addr to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)*
  %index.addr = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, align 8
  %index.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)** %index.addr to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)*
  %offset.addr = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, align 8
  %offset.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)** %offset.addr to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* %this, %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  store %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %extent, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %extent.addr.ascast, align 8, !tbaa !7
  store %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %index, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %index.addr.ascast, align 8, !tbaa !7
  store %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %offset, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %offset.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)*, %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %MImpl = getelementptr inbounds %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item", %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* %this1, i32 0, i32 0
  %MExtent = getelementptr inbounds %"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb1EEE.cl::sycl::detail::ItemBase", %"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb1EEE.cl::sycl::detail::ItemBase" addrspace(4)* %MImpl, i32 0, i32 0
  %0 = load %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %extent.addr.ascast, align 8, !tbaa !7
  %1 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %MExtent to i8 addrspace(4)*
  %2 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %0 to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %1, i8 addrspace(4)* align 8 %2, i64 16, i1 false)
  %MIndex = getelementptr inbounds %"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb1EEE.cl::sycl::detail::ItemBase", %"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb1EEE.cl::sycl::detail::ItemBase" addrspace(4)* %MImpl, i32 0, i32 1
  %3 = load %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %index.addr.ascast, align 8, !tbaa !7
  %4 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %MIndex to i8 addrspace(4)*
  %5 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %3 to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %4, i8 addrspace(4)* align 8 %5, i64 16, i1 false)
  %MOffset = getelementptr inbounds %"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb1EEE.cl::sycl::detail::ItemBase", %"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb1EEE.cl::sycl::detail::ItemBase" addrspace(4)* %MImpl, i32 0, i32 2
  %6 = load %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %offset.addr.ascast, align 8, !tbaa !7
  %7 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %MOffset to i8 addrspace(4)*
  %8 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %6 to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %7, i8 addrspace(4)* align 8 %8, i64 16, i1 false)
  ret void
}

; Function Attrs: convergent norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl4itemILi2ELb0EEC2ILb0EEERNSt9enable_ifIXntT_EKNS0_5rangeILi2EEEE4typeERKNS0_2idILi2EEE(%"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* align 8 dereferenceable_or_null(32) %this, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* align 8 dereferenceable(16) %extent, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* align 8 dereferenceable(16) %index) unnamed_addr #8 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* addrspace(4)*
  %extent.addr = alloca %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, align 8
  %extent.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)** %extent.addr to %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)*
  %index.addr = alloca %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, align 8
  %index.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)** %index.addr to %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* %this, %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  store %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %extent, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %extent.addr.ascast, align 8, !tbaa !7
  store %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %index, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %index.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)*, %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %MImpl = getelementptr inbounds %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item", %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* %this1, i32 0, i32 0
  %MExtent = getelementptr inbounds %"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb0EEE.cl::sycl::detail::ItemBase", %"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb0EEE.cl::sycl::detail::ItemBase" addrspace(4)* %MImpl, i32 0, i32 0
  %0 = load %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)*, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* addrspace(4)* %extent.addr.ascast, align 8, !tbaa !7
  %1 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %MExtent to i8 addrspace(4)*
  %2 = bitcast %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" addrspace(4)* %0 to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %1, i8 addrspace(4)* align 8 %2, i64 16, i1 false)
  %MIndex = getelementptr inbounds %"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb0EEE.cl::sycl::detail::ItemBase", %"struct._ZTSN2cl4sycl6detail8ItemBaseILi2ELb0EEE.cl::sycl::detail::ItemBase" addrspace(4)* %MImpl, i32 0, i32 1
  %3 = load %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)*, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* addrspace(4)* %index.addr.ascast, align 8, !tbaa !7
  %4 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %MIndex to i8 addrspace(4)*
  %5 = bitcast %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" addrspace(4)* %3 to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %4, i8 addrspace(4)* align 8 %5, i64 16, i1 false)
  ret void
}

; Function Attrs: convergent norecurse nounwind
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl7nd_itemILi2EEC2ERKNS0_4itemILi2ELb1EEERKNS3_ILi2ELb0EEERKNS0_5groupILi2EEE(%"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* align 8 dereferenceable_or_null(144) %this, %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* align 8 dereferenceable(48) %GL, %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* align 8 dereferenceable(32) %L, %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* align 8 dereferenceable(64) %GR) unnamed_addr #8 comdat align 2 {
entry:
  %this.addr = alloca %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)** %this.addr to %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* addrspace(4)*
  %GL.addr = alloca %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)*, align 8
  %GL.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)** %GL.addr to %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* addrspace(4)*
  %L.addr = alloca %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)*, align 8
  %L.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)** %L.addr to %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* addrspace(4)*
  %GR.addr = alloca %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)*, align 8
  %GR.addr.ascast = addrspacecast %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)** %GR.addr to %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* addrspace(4)*
  store %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* %this, %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8, !tbaa !7
  store %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* %GL, %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* addrspace(4)* %GL.addr.ascast, align 8, !tbaa !7
  store %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* %L, %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* addrspace(4)* %L.addr.ascast, align 8, !tbaa !7
  store %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* %GR, %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* addrspace(4)* %GR.addr.ascast, align 8, !tbaa !7
  %this1 = load %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)*, %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %globalItem = getelementptr inbounds %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item", %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* %this1, i32 0, i32 0
  %0 = load %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)*, %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* addrspace(4)* %GL.addr.ascast, align 8, !tbaa !7
  %1 = bitcast %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* %globalItem to i8 addrspace(4)*
  %2 = bitcast %"class._ZTSN2cl4sycl4itemILi2ELb1EEE.cl::sycl::item" addrspace(4)* %0 to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %1, i8 addrspace(4)* align 8 %2, i64 48, i1 false)
  %localItem = getelementptr inbounds %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item", %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* %this1, i32 0, i32 1
  %3 = load %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)*, %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* addrspace(4)* %L.addr.ascast, align 8, !tbaa !7
  %4 = bitcast %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* %localItem to i8 addrspace(4)*
  %5 = bitcast %"class._ZTSN2cl4sycl4itemILi2ELb0EEE.cl::sycl::item" addrspace(4)* %3 to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %4, i8 addrspace(4)* align 8 %5, i64 32, i1 false)
  %Group = getelementptr inbounds %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item", %"class._ZTSN2cl4sycl7nd_itemILi2EEE.cl::sycl::nd_item" addrspace(4)* %this1, i32 0, i32 2
  %6 = load %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)*, %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* addrspace(4)* %GR.addr.ascast, align 8, !tbaa !7
  %7 = bitcast %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* %Group to i8 addrspace(4)*
  %8 = bitcast %"class._ZTSN2cl4sycl5groupILi2EEE.cl::sycl::group" addrspace(4)* %6 to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %7, i8 addrspace(4)* align 8 %8, i64 64, i1 false)
  ret void
}

; Function Attrs: argmemonly nofree nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #4

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.experimental.noalias.scope.decl(metadata) #9

attributes #0 = { convergent norecurse mustprogress "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/export/users2/yubing/xmain0316/llvm/sycl/test/matrix/matrix-bf16-test.cpp" "uniform-work-group-size"="true" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent norecurse mustprogress "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { argmemonly nofree nounwind willreturn }
attributes #5 = { convergent inlinehint norecurse mustprogress "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #6 = { convergent norecurse nounwind mustprogress "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #7 = { argmemonly nofree nounwind willreturn writeonly }
attributes #8 = { convergent norecurse nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #9 = { inaccessiblememonly nofree nosync nounwind willreturn }
attributes #10 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #11 = { convergent inlinehint norecurse nounwind mustprogress "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/export/users2/yubing/xmain0316/llvm/sycl/test/matrix/matrix-bf16-test.cpp" }
attributes #12 = { nounwind }
attributes #13 = { convergent }
attributes #14 = { convergent nounwind }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!opencl.used.extensions = !{!4}
!opencl.used.optional.core.features = !{!4}
!opencl.compiler.options = !{!4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{}
!5 = !{!"Intel(R) oneAPI DPC++/C++ Compiler 2021.4.0 (2021.x.0.YYYYMMDD)"}
!6 = !{i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1}
!7 = !{!8, !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"long", !9, i64 0}
!13 = !{!14, !12, i64 168}
!14 = !{!"_ZTSZZ15matrix_multiplyIftLm94ELm292ELm146ELm188ELm94ELm94EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN2cl4sycl7handlerEE_clESC_EUlNSA_7nd_itemILi2EEEE_", !15, i64 0, !15, i64 56, !19, i64 112, !12, i64 168, !12, i64 176, !12, i64 184}
!15 = !{!"_ZTSN2cl4sycl8accessorItLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE", !16, i64 0, !9, i64 48}
!16 = !{!"_ZTSN2cl4sycl6detail18AccessorImplDeviceILi2EEE", !17, i64 0, !18, i64 16, !18, i64 32}
!17 = !{!"_ZTSN2cl4sycl2idILi2EEE"}
!18 = !{!"_ZTSN2cl4sycl5rangeILi2EEE"}
!19 = !{!"_ZTSN2cl4sycl8accessorIfLi2ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_6ONEAPI22accessor_property_listIJEEEEE", !16, i64 0, !9, i64 48}
!20 = !{!14, !12, i64 176}
!21 = !{!14, !12, i64 184}
!22 = !{!9, !9, i64 0}
!23 = !{!24, !24, i64 0}
!24 = !{!"int", !9, i64 0}
!25 = distinct !{!25, !26, !27}
!26 = !{!"llvm.loop.mustprogress"}
!27 = !{!"llvm.loop.unroll.enable"}
!28 = distinct !{!28, !26, !27}
!29 = !{!30, !30, i64 0}
!30 = !{!"_ZTSN2cl4sycl3ext5intel12experimental6matrix13matrix_layoutE", !9, i64 0}
!31 = !{!32, !8, i64 0}
!32 = !{!"_ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEfLm47ELm47ELNS4_13matrix_layoutE0EEE", !8, i64 0}
!33 = !{!34, !8, i64 0}
!34 = !{!"_ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm47ELm146ELNS4_13matrix_layoutE0EEE", !8, i64 0}
!35 = !{!36, !8, i64 0}
!36 = !{!"_ZTSN2cl4sycl3ext5intel12experimental6matrix12joint_matrixINS0_6ONEAPI9sub_groupEtLm146ELm47ELNS4_13matrix_layoutE3EEE", !8, i64 0}
!37 = !{!38}
!38 = distinct !{!38, !39, !"_ZN2cl4sycl3ext5intel12experimental6matrix16joint_matrix_madINS0_6ONEAPI9sub_groupEtfLm47ELm146ELm47ELNS4_13matrix_layoutE0ELS8_3ELS8_0EEENS4_12joint_matrixIT_T1_XT2_EXT4_EXT7_EEESA_RNS9_ISA_T0_XT2_EXT3_EXT5_EEERNS9_ISA_SD_XT3_EXT4_EXT6_EEERSC_: %agg.result"}
!39 = distinct !{!39, !"_ZN2cl4sycl3ext5intel12experimental6matrix16joint_matrix_madINS0_6ONEAPI9sub_groupEtfLm47ELm146ELm47ELNS4_13matrix_layoutE0ELS8_3ELS8_0EEENS4_12joint_matrixIT_T1_XT2_EXT4_EXT7_EEESA_RNS9_ISA_T0_XT2_EXT3_EXT5_EEERNS9_ISA_SD_XT3_EXT4_EXT6_EEERSC_"}
!40 = !{i64 0, i64 8, !7}
!41 = distinct !{!41, !26}
!42 = !{!43, !8, i64 0}
!43 = !{!"_ZTSN2cl4sycl9multi_ptrIfLNS0_6access13address_spaceE1EEE", !8, i64 0}
!44 = !{!45, !8, i64 0}
!45 = !{!"_ZTSN2cl4sycl9multi_ptrItLNS0_6access13address_spaceE1EEE", !8, i64 0}
!46 = distinct !{!46, !26}
!47 = distinct !{!47, !26}
