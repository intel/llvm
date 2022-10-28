; ModuleID = '/localdisk2/etiotto/intel-llvm/polygeist/tools/cgeist/Test/Verification/sycl/kernels.cpp'
source_filename = "/localdisk2/etiotto/intel-llvm/polygeist/tools/cgeist/Test/Verification/sycl/kernels.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown-syclmlir"

%"class.sycl::_V1::item.2.true" = type { %"struct.sycl::_V1::detail::ItemBase.2.true" }
%"struct.sycl::_V1::detail::ItemBase.2.true" = type { %"class.sycl::_V1::range.2", %"class.sycl::_V1::id.2", %"class.sycl::_V1::id.2" }
%"class.sycl::_V1::range.2" = type { %"class.sycl::_V1::detail::array.2" }
%"class.sycl::_V1::detail::array.2" = type { [2 x i64] }
%"class.sycl::_V1::id.2" = type { %"class.sycl::_V1::detail::array.2" }
%"class.sycl::_V1::range.1" = type { %"class.sycl::_V1::detail::array.1" }
%"class.sycl::_V1::detail::array.1" = type { [1 x i64] }
%"class.sycl::_V1::accessor.1" = type { %"class.sycl::_V1::detail::AccessorImplDevice.1", { i32 addrspace(1)* } }
%"class.sycl::_V1::detail::AccessorImplDevice.1" = type { %"class.sycl::_V1::id.1", %"class.sycl::_V1::range.1", %"class.sycl::_V1::range.1" }
%"class.sycl::_V1::id.1" = type { %"class.sycl::_V1::detail::array.1" }
%"class.sycl::_V1::item.1.true" = type { %"struct.sycl::_V1::detail::ItemBase.1.true" }
%"struct.sycl::_V1::detail::ItemBase.1.true" = type { %"class.sycl::_V1::range.1", %"class.sycl::_V1::id.1", %"class.sycl::_V1::id.1" }

@__spirv_BuiltInSubgroupLocalInvocationId = external global [1 x i32]
@__spirv_BuiltInSubgroupId = external global [1 x i32]
@__spirv_BuiltInNumSubgroups = external global [1 x i32]
@__spirv_BuiltInSubgroupMaxSize = external global [1 x i32]
@__spirv_BuiltInSubgroupSize = external global [1 x i32]
@__spirv_BuiltInLocalInvocationId = external global [3 x i64]
@__spirv_BuiltInWorkgroupId = external global [3 x i64]
@__spirv_BuiltInWorkgroupSize = external global [3 x i64]
@__spirv_BuiltInNumWorkgroups = external global [3 x i64]
@__spirv_BuiltInGlobalOffset = external global [3 x i64]
@__spirv_BuiltInGlobalSize = external global [3 x i64]
@__spirv_BuiltInGlobalInvocationId = external global [3 x i64]

declare i8* @malloc(i64)

declare void @free(i8*)

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z28__spirv_GlobalInvocationId_xv() #0 !dbg !3 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInGlobalInvocationId, i32 0, i32 0), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z28__spirv_GlobalInvocationId_yv() #0 !dbg !7 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInGlobalInvocationId, i32 0, i32 1), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z28__spirv_GlobalInvocationId_zv() #0 !dbg !8 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInGlobalInvocationId, i32 0, i32 2), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z20__spirv_GlobalSize_xv() #0 !dbg !9 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInGlobalSize, i32 0, i32 0), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z20__spirv_GlobalSize_yv() #0 !dbg !10 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInGlobalSize, i32 0, i32 1), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z20__spirv_GlobalSize_zv() #0 !dbg !11 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInGlobalSize, i32 0, i32 2), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z22__spirv_GlobalOffset_xv() #0 !dbg !12 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInGlobalOffset, i32 0, i32 0), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z22__spirv_GlobalOffset_yv() #0 !dbg !13 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInGlobalOffset, i32 0, i32 1), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z22__spirv_GlobalOffset_zv() #0 !dbg !14 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInGlobalOffset, i32 0, i32 2), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z23__spirv_NumWorkgroups_xv() #0 !dbg !15 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInNumWorkgroups, i32 0, i32 0), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z23__spirv_NumWorkgroups_yv() #0 !dbg !16 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInNumWorkgroups, i32 0, i32 1), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z23__spirv_NumWorkgroups_zv() #0 !dbg !17 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInNumWorkgroups, i32 0, i32 2), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z23__spirv_WorkgroupSize_xv() #0 !dbg !18 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInWorkgroupSize, i32 0, i32 0), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z23__spirv_WorkgroupSize_yv() #0 !dbg !19 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInWorkgroupSize, i32 0, i32 1), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z23__spirv_WorkgroupSize_zv() #0 !dbg !20 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInWorkgroupSize, i32 0, i32 2), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z21__spirv_WorkgroupId_xv() #0 !dbg !21 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInWorkgroupId, i32 0, i32 0), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z21__spirv_WorkgroupId_yv() #0 !dbg !22 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInWorkgroupId, i32 0, i32 1), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z21__spirv_WorkgroupId_zv() #0 !dbg !23 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInWorkgroupId, i32 0, i32 2), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z27__spirv_LocalInvocationId_xv() #0 !dbg !24 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInLocalInvocationId, i32 0, i32 0), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z27__spirv_LocalInvocationId_yv() #0 !dbg !25 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInLocalInvocationId, i32 0, i32 1), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i64 @_Z27__spirv_LocalInvocationId_zv() #0 !dbg !26 {
  %1 = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @__spirv_BuiltInLocalInvocationId, i32 0, i32 2), align 8
  ret i64 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i32 @_Z20__spirv_SubgroupSizev() #0 !dbg !27 {
  %1 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @__spirv_BuiltInSubgroupSize, i32 0, i32 0), align 4
  ret i32 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i32 @_Z23__spirv_SubgroupMaxSizev() #0 !dbg !28 {
  %1 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @__spirv_BuiltInSubgroupMaxSize, i32 0, i32 0), align 4
  ret i32 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i32 @_Z20__spirv_NumSubgroupsv() #0 !dbg !29 {
  %1 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @__spirv_BuiltInNumSubgroups, i32 0, i32 0), align 4
  ret i32 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i32 @_Z18__spirv_SubgroupIdv() #0 !dbg !30 {
  %1 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @__spirv_BuiltInSubgroupId, i32 0, i32 0), align 4
  ret i32 %1
}

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func i32 @_Z33__spirv_SubgroupLocalInvocationIdv() #0 !dbg !31 {
  %1 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @__spirv_BuiltInSubgroupLocalInvocationId, i32 0, i32 0), align 4
  ret i32 %1
}

; Function Attrs: convergent mustprogress norecurse nounwind
define spir_func void @_Z10function_1N4sycl3_V14itemILi2ELb1EEE(%"class.sycl::_V1::item.2.true"* %0) #1 !dbg !32 {
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
define spir_func void @_ZN4sycl3_V15rangeILi1EEC1ERKS2_(%"class.sycl::_V1::range.1" addrspace(4)* %0, %"class.sycl::_V1::range.1" addrspace(4)* %1) #1 {
  %3 = bitcast %"class.sycl::_V1::range.1" addrspace(4)* %0 to %"class.sycl::_V1::detail::array.1" addrspace(4)*
  %4 = bitcast %"class.sycl::_V1::range.1" addrspace(4)* %1 to %"class.sycl::_V1::detail::array.1" addrspace(4)*
  call void @_ZN4sycl3_V16detail5arrayILi1EEC1ERKS3_(%"class.sycl::_V1::detail::array.1" addrspace(4)* %3, %"class.sycl::_V1::detail::array.1" addrspace(4)* %4)
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
declare spir_func void @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC1Ev(%"class.sycl::_V1::accessor.1" addrspace(4)*) #1

; Function Attrs: convergent mustprogress norecurse nounwind
declare spir_func void @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_2idILi1EEE(%"class.sycl::_V1::accessor.1" addrspace(4)*, i32 addrspace(1)*, %"class.sycl::_V1::range.1"*, %"class.sycl::_V1::range.1"*, %"class.sycl::_V1::id.1"*) #1

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func void @_ZN4sycl3_V12idILi1EEC1ERKS2_(%"class.sycl::_V1::id.1" addrspace(4)* %0, %"class.sycl::_V1::id.1" addrspace(4)* %1) #0 {
  %3 = bitcast %"class.sycl::_V1::id.1" addrspace(4)* %0 to %"class.sycl::_V1::detail::array.1" addrspace(4)*
  %4 = bitcast %"class.sycl::_V1::id.1" addrspace(4)* %1 to %"class.sycl::_V1::detail::array.1" addrspace(4)*
  call void @_ZN4sycl3_V16detail5arrayILi1EEC1ERKS3_(%"class.sycl::_V1::detail::array.1" addrspace(4)* %3, %"class.sycl::_V1::detail::array.1" addrspace(4)* %4)
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
declare spir_func void @_ZNK4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1E8kernel_1EclES4_({ %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } } addrspace(4)*, %"class.sycl::_V1::item.1.true"*) #1

; Function Attrs: convergent mustprogress norecurse nounwind
declare spir_func %"class.sycl::_V1::item.1.true" @_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE(%"class.sycl::_V1::item.1.true" addrspace(4)*) #1

; Function Attrs: convergent mustprogress norecurse nounwind
declare spir_func %"class.sycl::_V1::item.1.true" addrspace(4)* @_ZN4sycl3_V16detail7declptrINS0_4itemILi1ELb1EEEEEPT_v() #1

; Function Attrs: convergent mustprogress norecurse nounwind
define spir_func void @_ZNK8kernel_1clEN4sycl3_V12idILi1EEE({ %"class.sycl::_V1::accessor.1" } addrspace(4)* %0, %"class.sycl::_V1::id.1"* %1) #1 {
  %3 = alloca %"class.sycl::_V1::id.1", align 8
  %4 = alloca %"class.sycl::_V1::id.1", align 8
  %5 = addrspacecast %"class.sycl::_V1::id.1"* %4 to %"class.sycl::_V1::id.1" addrspace(4)*
  %6 = addrspacecast %"class.sycl::_V1::id.1"* %1 to %"class.sycl::_V1::id.1" addrspace(4)*
  call void @_ZN4sycl3_V12idILi1EEC1ERKS2_(%"class.sycl::_V1::id.1" addrspace(4)* %5, %"class.sycl::_V1::id.1" addrspace(4)* %6)
  %7 = load %"class.sycl::_V1::id.1", %"class.sycl::_V1::id.1"* %4, align 8
  store %"class.sycl::_V1::id.1" %7, %"class.sycl::_V1::id.1"* %3, align 8
  %8 = bitcast { %"class.sycl::_V1::accessor.1" } addrspace(4)* %0 to %"class.sycl::_V1::accessor.1" addrspace(4)*
  %9 = call i32 addrspace(4)* @_ZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi1EvEERiNS0_2idILi1EEE(%"class.sycl::_V1::accessor.1" addrspace(4)* %8, %"class.sycl::_V1::id.1"* %3)
  store i32 42, i32 addrspace(4)* %9, align 4
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
declare spir_func void @_ZN4sycl3_V12idILi1EEC1ILi1ELb1EEERNSt9enable_ifIXeqT_Li1EEKNS0_4itemILi1EXT0_EEEE4typeE(%"class.sycl::_V1::id.1" addrspace(4)*, %"class.sycl::_V1::item.1.true" addrspace(4)*) #1

; Function Attrs: convergent mustprogress norecurse nounwind
declare spir_func void @_ZNK4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZZ6host_2vENKUlRNS0_7handlerEE_clES6_EUlNS0_2idILi1EEEE_EclES4_({ %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } } addrspace(4)*, %"class.sycl::_V1::item.1.true"*) #1

; Function Attrs: convergent inlinehint mustprogress norecurse nounwind
define spir_func void @_ZZZ6host_2vENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_2idILi1EEEE_clES5_({ %"class.sycl::_V1::accessor.1" } addrspace(4)* %0, %"class.sycl::_V1::id.1"* %1) #0 {
  %3 = alloca %"class.sycl::_V1::id.1", align 8
  %4 = alloca %"class.sycl::_V1::id.1", align 8
  %5 = addrspacecast %"class.sycl::_V1::id.1"* %4 to %"class.sycl::_V1::id.1" addrspace(4)*
  %6 = addrspacecast %"class.sycl::_V1::id.1"* %1 to %"class.sycl::_V1::id.1" addrspace(4)*
  call void @_ZN4sycl3_V12idILi1EEC1ERKS2_(%"class.sycl::_V1::id.1" addrspace(4)* %5, %"class.sycl::_V1::id.1" addrspace(4)* %6)
  %7 = load %"class.sycl::_V1::id.1", %"class.sycl::_V1::id.1"* %4, align 8
  store %"class.sycl::_V1::id.1" %7, %"class.sycl::_V1::id.1"* %3, align 8
  %8 = bitcast { %"class.sycl::_V1::accessor.1" } addrspace(4)* %0 to %"class.sycl::_V1::accessor.1" addrspace(4)*
  %9 = call i32 addrspace(4)* @_ZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi1EvEERiNS0_2idILi1EEE(%"class.sycl::_V1::accessor.1" addrspace(4)* %8, %"class.sycl::_V1::id.1"* %3)
  store i32 42, i32 addrspace(4)* %9, align 4
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
define spir_func void @_ZN4sycl3_V16detail5arrayILi1EEC1ERKS3_(%"class.sycl::_V1::detail::array.1" addrspace(4)* %0, %"class.sycl::_V1::detail::array.1" addrspace(4)* %1) #1 !dbg !34 {
  %3 = getelementptr %"class.sycl::_V1::detail::array.1", %"class.sycl::_V1::detail::array.1" addrspace(4)* %0, i32 0, i32 0, i32 0
  %4 = getelementptr %"class.sycl::_V1::detail::array.1", %"class.sycl::_V1::detail::array.1" addrspace(4)* %1, i32 0, i32 0, i32 0
  %5 = load i64, i64 addrspace(4)* %4, align 8
  store i64 %5, i64 addrspace(4)* %3, align 8
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
declare spir_func i32 addrspace(4)* @_ZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi1EvEERiNS0_2idILi1EEE(%"class.sycl::_V1::accessor.1" addrspace(4)*, %"class.sycl::_V1::id.1"*) #1

; Function Attrs: convergent mustprogress norecurse nounwind
define spir_kernel void @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1E8kernel_1EE(%"class.sycl::_V1::range.1"* noundef byval(%"class.sycl::_V1::range.1") align 8 %0, i32 addrspace(1)* %1, %"class.sycl::_V1::range.1"* noundef byval(%"class.sycl::_V1::range.1") align 8 %2, %"class.sycl::_V1::range.1"* noundef byval(%"class.sycl::_V1::range.1") align 8 %3, %"class.sycl::_V1::id.1"* noundef byval(%"class.sycl::_V1::id.1") align 8 %4) #1 {
  %6 = alloca %"class.sycl::_V1::item.1.true", align 8
  %7 = alloca %"class.sycl::_V1::id.1", align 8
  %8 = alloca %"class.sycl::_V1::range.1", align 8
  %9 = alloca %"class.sycl::_V1::range.1", align 8
  %10 = alloca %"class.sycl::_V1::id.1", align 8
  %11 = alloca %"class.sycl::_V1::range.1", align 8
  %12 = alloca %"class.sycl::_V1::range.1", align 8
  %13 = alloca %"class.sycl::_V1::accessor.1", align 8
  %14 = alloca %"class.sycl::_V1::range.1", align 8
  %15 = alloca { %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } }, align 8
  %16 = getelementptr { %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } }, { %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } }* %15, i32 0, i32 0
  %17 = addrspacecast %"class.sycl::_V1::range.1"* %14 to %"class.sycl::_V1::range.1" addrspace(4)*
  %18 = addrspacecast %"class.sycl::_V1::range.1"* %0 to %"class.sycl::_V1::range.1" addrspace(4)*
  call void @_ZN4sycl3_V15rangeILi1EEC1ERKS2_(%"class.sycl::_V1::range.1" addrspace(4)* %17, %"class.sycl::_V1::range.1" addrspace(4)* %18)
  %19 = load %"class.sycl::_V1::range.1", %"class.sycl::_V1::range.1"* %14, align 8
  store %"class.sycl::_V1::range.1" %19, %"class.sycl::_V1::range.1"* %16, align 8
  %20 = getelementptr { %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } }, { %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } }* %15, i32 0, i32 1
  %21 = getelementptr { %"class.sycl::_V1::accessor.1" }, { %"class.sycl::_V1::accessor.1" }* %20, i32 0, i32 0
  %22 = addrspacecast %"class.sycl::_V1::accessor.1"* %13 to %"class.sycl::_V1::accessor.1" addrspace(4)*
  call void @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC1Ev(%"class.sycl::_V1::accessor.1" addrspace(4)* %22)
  %23 = load %"class.sycl::_V1::accessor.1", %"class.sycl::_V1::accessor.1"* %13, align 8
  store %"class.sycl::_V1::accessor.1" %23, %"class.sycl::_V1::accessor.1"* %21, align 8
  %24 = addrspacecast %"class.sycl::_V1::range.1"* %12 to %"class.sycl::_V1::range.1" addrspace(4)*
  %25 = addrspacecast %"class.sycl::_V1::range.1"* %2 to %"class.sycl::_V1::range.1" addrspace(4)*
  call void @_ZN4sycl3_V15rangeILi1EEC1ERKS2_(%"class.sycl::_V1::range.1" addrspace(4)* %24, %"class.sycl::_V1::range.1" addrspace(4)* %25)
  %26 = addrspacecast %"class.sycl::_V1::range.1"* %11 to %"class.sycl::_V1::range.1" addrspace(4)*
  %27 = addrspacecast %"class.sycl::_V1::range.1"* %3 to %"class.sycl::_V1::range.1" addrspace(4)*
  call void @_ZN4sycl3_V15rangeILi1EEC1ERKS2_(%"class.sycl::_V1::range.1" addrspace(4)* %26, %"class.sycl::_V1::range.1" addrspace(4)* %27)
  %28 = addrspacecast %"class.sycl::_V1::id.1"* %10 to %"class.sycl::_V1::id.1" addrspace(4)*
  %29 = addrspacecast %"class.sycl::_V1::id.1"* %4 to %"class.sycl::_V1::id.1" addrspace(4)*
  call void @_ZN4sycl3_V12idILi1EEC1ERKS2_(%"class.sycl::_V1::id.1" addrspace(4)* %28, %"class.sycl::_V1::id.1" addrspace(4)* %29)
  %30 = bitcast { %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } }* %15 to %"class.sycl::_V1::accessor.1"*
  %31 = getelementptr %"class.sycl::_V1::accessor.1", %"class.sycl::_V1::accessor.1"* %30, i64 1
  %32 = addrspacecast %"class.sycl::_V1::accessor.1"* %31 to %"class.sycl::_V1::accessor.1" addrspace(4)*
  %33 = load %"class.sycl::_V1::range.1", %"class.sycl::_V1::range.1"* %12, align 8
  store %"class.sycl::_V1::range.1" %33, %"class.sycl::_V1::range.1"* %9, align 8
  %34 = load %"class.sycl::_V1::range.1", %"class.sycl::_V1::range.1"* %11, align 8
  store %"class.sycl::_V1::range.1" %34, %"class.sycl::_V1::range.1"* %8, align 8
  %35 = load %"class.sycl::_V1::id.1", %"class.sycl::_V1::id.1"* %10, align 8
  store %"class.sycl::_V1::id.1" %35, %"class.sycl::_V1::id.1"* %7, align 8
  call void @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_2idILi1EEE(%"class.sycl::_V1::accessor.1" addrspace(4)* %32, i32 addrspace(1)* %1, %"class.sycl::_V1::range.1"* %9, %"class.sycl::_V1::range.1"* %8, %"class.sycl::_V1::id.1"* %7)
  %36 = call %"class.sycl::_V1::item.1.true" addrspace(4)* @_ZN4sycl3_V16detail7declptrINS0_4itemILi1ELb1EEEEEPT_v()
  %37 = call %"class.sycl::_V1::item.1.true" @_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE(%"class.sycl::_V1::item.1.true" addrspace(4)* %36)
  %38 = addrspacecast { %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } }* %15 to { %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } } addrspace(4)*
  store %"class.sycl::_V1::item.1.true" %37, %"class.sycl::_V1::item.1.true"* %6, align 8
  call void @_ZNK4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1E8kernel_1EclES4_({ %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } } addrspace(4)* %38, %"class.sycl::_V1::item.1.true"* %6)
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
define spir_kernel void @_ZTS8kernel_1(i32 addrspace(1)* %0, %"class.sycl::_V1::range.1"* noundef byval(%"class.sycl::_V1::range.1") align 8 %1, %"class.sycl::_V1::range.1"* noundef byval(%"class.sycl::_V1::range.1") align 8 %2, %"class.sycl::_V1::id.1"* noundef byval(%"class.sycl::_V1::id.1") align 8 %3) #1 {
  %5 = alloca %"class.sycl::_V1::id.1", align 8
  %6 = alloca %"class.sycl::_V1::item.1.true", align 8
  %7 = alloca %"class.sycl::_V1::id.1", align 8
  %8 = alloca %"class.sycl::_V1::id.1", align 8
  %9 = alloca %"class.sycl::_V1::range.1", align 8
  %10 = alloca %"class.sycl::_V1::range.1", align 8
  %11 = alloca %"class.sycl::_V1::id.1", align 8
  %12 = alloca %"class.sycl::_V1::range.1", align 8
  %13 = alloca %"class.sycl::_V1::range.1", align 8
  %14 = alloca %"class.sycl::_V1::accessor.1", align 8
  %15 = alloca { %"class.sycl::_V1::accessor.1" }, align 8
  %16 = getelementptr { %"class.sycl::_V1::accessor.1" }, { %"class.sycl::_V1::accessor.1" }* %15, i32 0, i32 0
  %17 = addrspacecast %"class.sycl::_V1::accessor.1"* %14 to %"class.sycl::_V1::accessor.1" addrspace(4)*
  call void @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC1Ev(%"class.sycl::_V1::accessor.1" addrspace(4)* %17)
  %18 = load %"class.sycl::_V1::accessor.1", %"class.sycl::_V1::accessor.1"* %14, align 8
  store %"class.sycl::_V1::accessor.1" %18, %"class.sycl::_V1::accessor.1"* %16, align 8
  %19 = addrspacecast %"class.sycl::_V1::range.1"* %13 to %"class.sycl::_V1::range.1" addrspace(4)*
  %20 = addrspacecast %"class.sycl::_V1::range.1"* %1 to %"class.sycl::_V1::range.1" addrspace(4)*
  call void @_ZN4sycl3_V15rangeILi1EEC1ERKS2_(%"class.sycl::_V1::range.1" addrspace(4)* %19, %"class.sycl::_V1::range.1" addrspace(4)* %20)
  %21 = addrspacecast %"class.sycl::_V1::range.1"* %12 to %"class.sycl::_V1::range.1" addrspace(4)*
  %22 = addrspacecast %"class.sycl::_V1::range.1"* %2 to %"class.sycl::_V1::range.1" addrspace(4)*
  call void @_ZN4sycl3_V15rangeILi1EEC1ERKS2_(%"class.sycl::_V1::range.1" addrspace(4)* %21, %"class.sycl::_V1::range.1" addrspace(4)* %22)
  %23 = addrspacecast %"class.sycl::_V1::id.1"* %11 to %"class.sycl::_V1::id.1" addrspace(4)*
  %24 = addrspacecast %"class.sycl::_V1::id.1"* %3 to %"class.sycl::_V1::id.1" addrspace(4)*
  call void @_ZN4sycl3_V12idILi1EEC1ERKS2_(%"class.sycl::_V1::id.1" addrspace(4)* %23, %"class.sycl::_V1::id.1" addrspace(4)* %24)
  %25 = bitcast { %"class.sycl::_V1::accessor.1" }* %15 to %"class.sycl::_V1::accessor.1"*
  %26 = addrspacecast %"class.sycl::_V1::accessor.1"* %25 to %"class.sycl::_V1::accessor.1" addrspace(4)*
  %27 = load %"class.sycl::_V1::range.1", %"class.sycl::_V1::range.1"* %13, align 8
  store %"class.sycl::_V1::range.1" %27, %"class.sycl::_V1::range.1"* %10, align 8
  %28 = load %"class.sycl::_V1::range.1", %"class.sycl::_V1::range.1"* %12, align 8
  store %"class.sycl::_V1::range.1" %28, %"class.sycl::_V1::range.1"* %9, align 8
  %29 = load %"class.sycl::_V1::id.1", %"class.sycl::_V1::id.1"* %11, align 8
  store %"class.sycl::_V1::id.1" %29, %"class.sycl::_V1::id.1"* %8, align 8
  call void @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_2idILi1EEE(%"class.sycl::_V1::accessor.1" addrspace(4)* %26, i32 addrspace(1)* %0, %"class.sycl::_V1::range.1"* %10, %"class.sycl::_V1::range.1"* %9, %"class.sycl::_V1::id.1"* %8)
  %30 = call %"class.sycl::_V1::item.1.true" addrspace(4)* @_ZN4sycl3_V16detail7declptrINS0_4itemILi1ELb1EEEEEPT_v()
  %31 = call %"class.sycl::_V1::item.1.true" @_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE(%"class.sycl::_V1::item.1.true" addrspace(4)* %30)
  store %"class.sycl::_V1::item.1.true" %31, %"class.sycl::_V1::item.1.true"* %6, align 8
  %32 = addrspacecast %"class.sycl::_V1::id.1"* %7 to %"class.sycl::_V1::id.1" addrspace(4)*
  %33 = addrspacecast %"class.sycl::_V1::item.1.true"* %6 to %"class.sycl::_V1::item.1.true" addrspace(4)*
  call void @_ZN4sycl3_V12idILi1EEC1ILi1ELb1EEERNSt9enable_ifIXeqT_Li1EEKNS0_4itemILi1EXT0_EEEE4typeE(%"class.sycl::_V1::id.1" addrspace(4)* %32, %"class.sycl::_V1::item.1.true" addrspace(4)* %33)
  %34 = addrspacecast { %"class.sycl::_V1::accessor.1" }* %15 to { %"class.sycl::_V1::accessor.1" } addrspace(4)*
  %35 = load %"class.sycl::_V1::id.1", %"class.sycl::_V1::id.1"* %7, align 8
  store %"class.sycl::_V1::id.1" %35, %"class.sycl::_V1::id.1"* %5, align 8
  call void @_ZNK8kernel_1clEN4sycl3_V12idILi1EEE({ %"class.sycl::_V1::accessor.1" } addrspace(4)* %34, %"class.sycl::_V1::id.1"* %5)
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
define spir_kernel void @_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZ6host_2vENKUlRNS0_7handlerEE_clES4_E8kernel_2EE(%"class.sycl::_V1::range.1"* noundef byval(%"class.sycl::_V1::range.1") align 8 %0, i32 addrspace(1)* %1, %"class.sycl::_V1::range.1"* noundef byval(%"class.sycl::_V1::range.1") align 8 %2, %"class.sycl::_V1::range.1"* noundef byval(%"class.sycl::_V1::range.1") align 8 %3, %"class.sycl::_V1::id.1"* noundef byval(%"class.sycl::_V1::id.1") align 8 %4) #1 {
  %6 = alloca %"class.sycl::_V1::item.1.true", align 8
  %7 = alloca %"class.sycl::_V1::id.1", align 8
  %8 = alloca %"class.sycl::_V1::range.1", align 8
  %9 = alloca %"class.sycl::_V1::range.1", align 8
  %10 = alloca %"class.sycl::_V1::id.1", align 8
  %11 = alloca %"class.sycl::_V1::range.1", align 8
  %12 = alloca %"class.sycl::_V1::range.1", align 8
  %13 = alloca %"class.sycl::_V1::accessor.1", align 8
  %14 = alloca %"class.sycl::_V1::range.1", align 8
  %15 = alloca { %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } }, align 8
  %16 = getelementptr { %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } }, { %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } }* %15, i32 0, i32 0
  %17 = addrspacecast %"class.sycl::_V1::range.1"* %14 to %"class.sycl::_V1::range.1" addrspace(4)*
  %18 = addrspacecast %"class.sycl::_V1::range.1"* %0 to %"class.sycl::_V1::range.1" addrspace(4)*
  call void @_ZN4sycl3_V15rangeILi1EEC1ERKS2_(%"class.sycl::_V1::range.1" addrspace(4)* %17, %"class.sycl::_V1::range.1" addrspace(4)* %18)
  %19 = load %"class.sycl::_V1::range.1", %"class.sycl::_V1::range.1"* %14, align 8
  store %"class.sycl::_V1::range.1" %19, %"class.sycl::_V1::range.1"* %16, align 8
  %20 = getelementptr { %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } }, { %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } }* %15, i32 0, i32 1
  %21 = getelementptr { %"class.sycl::_V1::accessor.1" }, { %"class.sycl::_V1::accessor.1" }* %20, i32 0, i32 0
  %22 = addrspacecast %"class.sycl::_V1::accessor.1"* %13 to %"class.sycl::_V1::accessor.1" addrspace(4)*
  call void @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC1Ev(%"class.sycl::_V1::accessor.1" addrspace(4)* %22)
  %23 = load %"class.sycl::_V1::accessor.1", %"class.sycl::_V1::accessor.1"* %13, align 8
  store %"class.sycl::_V1::accessor.1" %23, %"class.sycl::_V1::accessor.1"* %21, align 8
  %24 = addrspacecast %"class.sycl::_V1::range.1"* %12 to %"class.sycl::_V1::range.1" addrspace(4)*
  %25 = addrspacecast %"class.sycl::_V1::range.1"* %2 to %"class.sycl::_V1::range.1" addrspace(4)*
  call void @_ZN4sycl3_V15rangeILi1EEC1ERKS2_(%"class.sycl::_V1::range.1" addrspace(4)* %24, %"class.sycl::_V1::range.1" addrspace(4)* %25)
  %26 = addrspacecast %"class.sycl::_V1::range.1"* %11 to %"class.sycl::_V1::range.1" addrspace(4)*
  %27 = addrspacecast %"class.sycl::_V1::range.1"* %3 to %"class.sycl::_V1::range.1" addrspace(4)*
  call void @_ZN4sycl3_V15rangeILi1EEC1ERKS2_(%"class.sycl::_V1::range.1" addrspace(4)* %26, %"class.sycl::_V1::range.1" addrspace(4)* %27)
  %28 = addrspacecast %"class.sycl::_V1::id.1"* %10 to %"class.sycl::_V1::id.1" addrspace(4)*
  %29 = addrspacecast %"class.sycl::_V1::id.1"* %4 to %"class.sycl::_V1::id.1" addrspace(4)*
  call void @_ZN4sycl3_V12idILi1EEC1ERKS2_(%"class.sycl::_V1::id.1" addrspace(4)* %28, %"class.sycl::_V1::id.1" addrspace(4)* %29)
  %30 = bitcast { %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } }* %15 to %"class.sycl::_V1::accessor.1"*
  %31 = getelementptr %"class.sycl::_V1::accessor.1", %"class.sycl::_V1::accessor.1"* %30, i64 1
  %32 = addrspacecast %"class.sycl::_V1::accessor.1"* %31 to %"class.sycl::_V1::accessor.1" addrspace(4)*
  %33 = load %"class.sycl::_V1::range.1", %"class.sycl::_V1::range.1"* %12, align 8
  store %"class.sycl::_V1::range.1" %33, %"class.sycl::_V1::range.1"* %9, align 8
  %34 = load %"class.sycl::_V1::range.1", %"class.sycl::_V1::range.1"* %11, align 8
  store %"class.sycl::_V1::range.1" %34, %"class.sycl::_V1::range.1"* %8, align 8
  %35 = load %"class.sycl::_V1::id.1", %"class.sycl::_V1::id.1"* %10, align 8
  store %"class.sycl::_V1::id.1" %35, %"class.sycl::_V1::id.1"* %7, align 8
  call void @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_2idILi1EEE(%"class.sycl::_V1::accessor.1" addrspace(4)* %32, i32 addrspace(1)* %1, %"class.sycl::_V1::range.1"* %9, %"class.sycl::_V1::range.1"* %8, %"class.sycl::_V1::id.1"* %7)
  %36 = call %"class.sycl::_V1::item.1.true" addrspace(4)* @_ZN4sycl3_V16detail7declptrINS0_4itemILi1ELb1EEEEEPT_v()
  %37 = call %"class.sycl::_V1::item.1.true" @_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE(%"class.sycl::_V1::item.1.true" addrspace(4)* %36)
  %38 = addrspacecast { %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } }* %15 to { %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } } addrspace(4)*
  store %"class.sycl::_V1::item.1.true" %37, %"class.sycl::_V1::item.1.true"* %6, align 8
  call void @_ZNK4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZZ6host_2vENKUlRNS0_7handlerEE_clES6_EUlNS0_2idILi1EEEE_EclES4_({ %"class.sycl::_V1::range.1", { %"class.sycl::_V1::accessor.1" } } addrspace(4)* %38, %"class.sycl::_V1::item.1.true"* %6)
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
define spir_kernel void @_ZTSZZ6host_2vENKUlRN4sycl3_V17handlerEE_clES2_E8kernel_2(i32 addrspace(1)* %0, %"class.sycl::_V1::range.1"* noundef byval(%"class.sycl::_V1::range.1") align 8 %1, %"class.sycl::_V1::range.1"* noundef byval(%"class.sycl::_V1::range.1") align 8 %2, %"class.sycl::_V1::id.1"* noundef byval(%"class.sycl::_V1::id.1") align 8 %3) #1 {
  %5 = alloca %"class.sycl::_V1::id.1", align 8
  %6 = alloca %"class.sycl::_V1::item.1.true", align 8
  %7 = alloca %"class.sycl::_V1::id.1", align 8
  %8 = alloca %"class.sycl::_V1::id.1", align 8
  %9 = alloca %"class.sycl::_V1::range.1", align 8
  %10 = alloca %"class.sycl::_V1::range.1", align 8
  %11 = alloca %"class.sycl::_V1::id.1", align 8
  %12 = alloca %"class.sycl::_V1::range.1", align 8
  %13 = alloca %"class.sycl::_V1::range.1", align 8
  %14 = alloca %"class.sycl::_V1::accessor.1", align 8
  %15 = alloca { %"class.sycl::_V1::accessor.1" }, align 8
  %16 = getelementptr { %"class.sycl::_V1::accessor.1" }, { %"class.sycl::_V1::accessor.1" }* %15, i32 0, i32 0
  %17 = addrspacecast %"class.sycl::_V1::accessor.1"* %14 to %"class.sycl::_V1::accessor.1" addrspace(4)*
  call void @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC1Ev(%"class.sycl::_V1::accessor.1" addrspace(4)* %17)
  %18 = load %"class.sycl::_V1::accessor.1", %"class.sycl::_V1::accessor.1"* %14, align 8
  store %"class.sycl::_V1::accessor.1" %18, %"class.sycl::_V1::accessor.1"* %16, align 8
  %19 = addrspacecast %"class.sycl::_V1::range.1"* %13 to %"class.sycl::_V1::range.1" addrspace(4)*
  %20 = addrspacecast %"class.sycl::_V1::range.1"* %1 to %"class.sycl::_V1::range.1" addrspace(4)*
  call void @_ZN4sycl3_V15rangeILi1EEC1ERKS2_(%"class.sycl::_V1::range.1" addrspace(4)* %19, %"class.sycl::_V1::range.1" addrspace(4)* %20)
  %21 = addrspacecast %"class.sycl::_V1::range.1"* %12 to %"class.sycl::_V1::range.1" addrspace(4)*
  %22 = addrspacecast %"class.sycl::_V1::range.1"* %2 to %"class.sycl::_V1::range.1" addrspace(4)*
  call void @_ZN4sycl3_V15rangeILi1EEC1ERKS2_(%"class.sycl::_V1::range.1" addrspace(4)* %21, %"class.sycl::_V1::range.1" addrspace(4)* %22)
  %23 = addrspacecast %"class.sycl::_V1::id.1"* %11 to %"class.sycl::_V1::id.1" addrspace(4)*
  %24 = addrspacecast %"class.sycl::_V1::id.1"* %3 to %"class.sycl::_V1::id.1" addrspace(4)*
  call void @_ZN4sycl3_V12idILi1EEC1ERKS2_(%"class.sycl::_V1::id.1" addrspace(4)* %23, %"class.sycl::_V1::id.1" addrspace(4)* %24)
  %25 = bitcast { %"class.sycl::_V1::accessor.1" }* %15 to %"class.sycl::_V1::accessor.1"*
  %26 = addrspacecast %"class.sycl::_V1::accessor.1"* %25 to %"class.sycl::_V1::accessor.1" addrspace(4)*
  %27 = load %"class.sycl::_V1::range.1", %"class.sycl::_V1::range.1"* %13, align 8
  store %"class.sycl::_V1::range.1" %27, %"class.sycl::_V1::range.1"* %10, align 8
  %28 = load %"class.sycl::_V1::range.1", %"class.sycl::_V1::range.1"* %12, align 8
  store %"class.sycl::_V1::range.1" %28, %"class.sycl::_V1::range.1"* %9, align 8
  %29 = load %"class.sycl::_V1::id.1", %"class.sycl::_V1::id.1"* %11, align 8
  store %"class.sycl::_V1::id.1" %29, %"class.sycl::_V1::id.1"* %8, align 8
  call void @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_2idILi1EEE(%"class.sycl::_V1::accessor.1" addrspace(4)* %26, i32 addrspace(1)* %0, %"class.sycl::_V1::range.1"* %10, %"class.sycl::_V1::range.1"* %9, %"class.sycl::_V1::id.1"* %8)
  %30 = call %"class.sycl::_V1::item.1.true" addrspace(4)* @_ZN4sycl3_V16detail7declptrINS0_4itemILi1ELb1EEEEEPT_v()
  %31 = call %"class.sycl::_V1::item.1.true" @_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE(%"class.sycl::_V1::item.1.true" addrspace(4)* %30)
  store %"class.sycl::_V1::item.1.true" %31, %"class.sycl::_V1::item.1.true"* %6, align 8
  %32 = addrspacecast %"class.sycl::_V1::id.1"* %7 to %"class.sycl::_V1::id.1" addrspace(4)*
  %33 = addrspacecast %"class.sycl::_V1::item.1.true"* %6 to %"class.sycl::_V1::item.1.true" addrspace(4)*
  call void @_ZN4sycl3_V12idILi1EEC1ILi1ELb1EEERNSt9enable_ifIXeqT_Li1EEKNS0_4itemILi1EXT0_EEEE4typeE(%"class.sycl::_V1::id.1" addrspace(4)* %32, %"class.sycl::_V1::item.1.true" addrspace(4)* %33)
  %34 = addrspacecast { %"class.sycl::_V1::accessor.1" }* %15 to { %"class.sycl::_V1::accessor.1" } addrspace(4)*
  %35 = load %"class.sycl::_V1::id.1", %"class.sycl::_V1::id.1"* %7, align 8
  store %"class.sycl::_V1::id.1" %35, %"class.sycl::_V1::id.1"* %5, align 8
  call void @_ZZZ6host_2vENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_2idILi1EEEE_clES5_({ %"class.sycl::_V1::accessor.1" } addrspace(4)* %34, %"class.sycl::_V1::id.1"* %5)
  ret void
}

attributes #0 = { convergent inlinehint mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/localdisk2/etiotto/intel-llvm/polygeist/tools/cgeist/Test/Verification/sycl/kernels.cpp" }
attributes #1 = { convergent mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/localdisk2/etiotto/intel-llvm/polygeist/tools/cgeist/Test/Verification/sycl/kernels.cpp" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "/localdisk2/etiotto/intel-llvm/polygeist/tools/cgeist/Test/Verification/sycl/kernels.cpp", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "_Z28__spirv_GlobalInvocationId_xv", linkageName: "_Z28__spirv_GlobalInvocationId_xv", scope: null, file: !4, line: 71, type: !5, scopeLine: 71, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "/localdisk2/etiotto/intel-llvm/build/bin/../include/sycl/CL/__spirv/spirv_vars.hpp", directory: "")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = distinct !DISubprogram(name: "_Z28__spirv_GlobalInvocationId_yv", linkageName: "_Z28__spirv_GlobalInvocationId_yv", scope: null, file: !4, line: 74, type: !5, scopeLine: 74, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!8 = distinct !DISubprogram(name: "_Z28__spirv_GlobalInvocationId_zv", linkageName: "_Z28__spirv_GlobalInvocationId_zv", scope: null, file: !4, line: 77, type: !5, scopeLine: 77, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!9 = distinct !DISubprogram(name: "_Z20__spirv_GlobalSize_xv", linkageName: "_Z20__spirv_GlobalSize_xv", scope: null, file: !4, line: 81, type: !5, scopeLine: 81, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!10 = distinct !DISubprogram(name: "_Z20__spirv_GlobalSize_yv", linkageName: "_Z20__spirv_GlobalSize_yv", scope: null, file: !4, line: 84, type: !5, scopeLine: 84, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!11 = distinct !DISubprogram(name: "_Z20__spirv_GlobalSize_zv", linkageName: "_Z20__spirv_GlobalSize_zv", scope: null, file: !4, line: 87, type: !5, scopeLine: 87, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!12 = distinct !DISubprogram(name: "_Z22__spirv_GlobalOffset_xv", linkageName: "_Z22__spirv_GlobalOffset_xv", scope: null, file: !4, line: 91, type: !5, scopeLine: 91, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!13 = distinct !DISubprogram(name: "_Z22__spirv_GlobalOffset_yv", linkageName: "_Z22__spirv_GlobalOffset_yv", scope: null, file: !4, line: 94, type: !5, scopeLine: 94, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!14 = distinct !DISubprogram(name: "_Z22__spirv_GlobalOffset_zv", linkageName: "_Z22__spirv_GlobalOffset_zv", scope: null, file: !4, line: 97, type: !5, scopeLine: 97, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!15 = distinct !DISubprogram(name: "_Z23__spirv_NumWorkgroups_xv", linkageName: "_Z23__spirv_NumWorkgroups_xv", scope: null, file: !4, line: 101, type: !5, scopeLine: 101, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!16 = distinct !DISubprogram(name: "_Z23__spirv_NumWorkgroups_yv", linkageName: "_Z23__spirv_NumWorkgroups_yv", scope: null, file: !4, line: 104, type: !5, scopeLine: 104, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!17 = distinct !DISubprogram(name: "_Z23__spirv_NumWorkgroups_zv", linkageName: "_Z23__spirv_NumWorkgroups_zv", scope: null, file: !4, line: 107, type: !5, scopeLine: 107, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!18 = distinct !DISubprogram(name: "_Z23__spirv_WorkgroupSize_xv", linkageName: "_Z23__spirv_WorkgroupSize_xv", scope: null, file: !4, line: 111, type: !5, scopeLine: 111, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!19 = distinct !DISubprogram(name: "_Z23__spirv_WorkgroupSize_yv", linkageName: "_Z23__spirv_WorkgroupSize_yv", scope: null, file: !4, line: 114, type: !5, scopeLine: 114, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!20 = distinct !DISubprogram(name: "_Z23__spirv_WorkgroupSize_zv", linkageName: "_Z23__spirv_WorkgroupSize_zv", scope: null, file: !4, line: 117, type: !5, scopeLine: 117, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!21 = distinct !DISubprogram(name: "_Z21__spirv_WorkgroupId_xv", linkageName: "_Z21__spirv_WorkgroupId_xv", scope: null, file: !4, line: 121, type: !5, scopeLine: 121, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!22 = distinct !DISubprogram(name: "_Z21__spirv_WorkgroupId_yv", linkageName: "_Z21__spirv_WorkgroupId_yv", scope: null, file: !4, line: 124, type: !5, scopeLine: 124, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!23 = distinct !DISubprogram(name: "_Z21__spirv_WorkgroupId_zv", linkageName: "_Z21__spirv_WorkgroupId_zv", scope: null, file: !4, line: 127, type: !5, scopeLine: 127, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!24 = distinct !DISubprogram(name: "_Z27__spirv_LocalInvocationId_xv", linkageName: "_Z27__spirv_LocalInvocationId_xv", scope: null, file: !4, line: 131, type: !5, scopeLine: 131, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!25 = distinct !DISubprogram(name: "_Z27__spirv_LocalInvocationId_yv", linkageName: "_Z27__spirv_LocalInvocationId_yv", scope: null, file: !4, line: 134, type: !5, scopeLine: 134, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!26 = distinct !DISubprogram(name: "_Z27__spirv_LocalInvocationId_zv", linkageName: "_Z27__spirv_LocalInvocationId_zv", scope: null, file: !4, line: 137, type: !5, scopeLine: 137, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!27 = distinct !DISubprogram(name: "_Z20__spirv_SubgroupSizev", linkageName: "_Z20__spirv_SubgroupSizev", scope: null, file: !4, line: 141, type: !5, scopeLine: 141, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!28 = distinct !DISubprogram(name: "_Z23__spirv_SubgroupMaxSizev", linkageName: "_Z23__spirv_SubgroupMaxSizev", scope: null, file: !4, line: 144, type: !5, scopeLine: 144, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!29 = distinct !DISubprogram(name: "_Z20__spirv_NumSubgroupsv", linkageName: "_Z20__spirv_NumSubgroupsv", scope: null, file: !4, line: 147, type: !5, scopeLine: 147, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!30 = distinct !DISubprogram(name: "_Z18__spirv_SubgroupIdv", linkageName: "_Z18__spirv_SubgroupIdv", scope: null, file: !4, line: 150, type: !5, scopeLine: 150, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!31 = distinct !DISubprogram(name: "_Z33__spirv_SubgroupLocalInvocationIdv", linkageName: "_Z33__spirv_SubgroupLocalInvocationIdv", scope: null, file: !4, line: 153, type: !5, scopeLine: 153, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!32 = distinct !DISubprogram(name: "_Z10function_1N4sycl3_V14itemILi2ELb1EEE", linkageName: "_Z10function_1N4sycl3_V14itemILi2ELb1EEE", scope: null, file: !33, line: 80, type: !5, scopeLine: 80, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!33 = !DIFile(filename: "/localdisk2/etiotto/intel-llvm/polygeist/tools/cgeist/Test/Verification/sycl/kernels.cpp", directory: "")
!34 = distinct !DISubprogram(name: "_ZN4sycl3_V16detail5arrayILi1EEC1ERKS3_", linkageName: "_ZN4sycl3_V16detail5arrayILi1EEC1ERKS3_", scope: null, file: !35, line: 82, type: !5, scopeLine: 82, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!35 = !DIFile(filename: "/localdisk2/etiotto/intel-llvm/build/bin/../include/sycl/detail/array.hpp", directory: "")

