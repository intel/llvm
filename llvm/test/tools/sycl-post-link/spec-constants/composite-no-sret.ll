; RUN: sycl-post-link -spec-const=rt --ir-output-only %s -S -o - \
; RUN: | FileCheck %s --implicit-check-not "call {{.*}} __sycl_getCompositeSpecConstantValue" --implicit-check-not "call {{.*}} __sycl_getComposite2020SpecConstantValue"

; CHECK: %[[#NS0:]] = call i32 @_Z20__spirv_SpecConstantii(i32 [[#ID:]], i32
; CHECK: %[[#NS1:]] = call i32 @_Z20__spirv_SpecConstantii(i32 [[#ID + 1]], i32 42)
; CHECK: %[[#NA0:]] = call %struct._ZTS10TestStruct.TestStruct @_Z29__spirv_SpecConstantCompositeii(i32 %[[#NS0]], i32 %[[#NS1]])

; CHECK: declare i32 @_Z20__spirv_SpecConstantii(i32, i32)
; CHECK: declare %struct._ZTS10TestStruct.TestStruct @_Z29__spirv_SpecConstantCompositeii(i32, i32)

; CHECK: !sycl.specialization-constants = !{![[#MD:]]}
; CHECK: ![[#MD]] = !{!"_ZTSN2cl4sycl6detail32specialization_id_name_generatorIL_ZL10SpecConst3EEE", i32 [[#ID]], i32 0, i32 4,

; ModuleID = 'cuda.mod.bc'
source_filename = "common.cpp"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown-sycldevice"

%"class._ZTSN2cl4sycl17specialization_idIiEE.cl::sycl::specialization_id" = type { i32 }
%"class._ZTSN2cl4sycl17specialization_idI10TestStructEE.cl::sycl::specialization_id" = type { %struct._ZTS10TestStruct.TestStruct }
%struct._ZTS10TestStruct.TestStruct = type { i32, i32 }
%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }
%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }

$"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE11Kernel4Name" = comdat any

@_ZL10SpecConst2 = internal constant %"class._ZTSN2cl4sycl17specialization_idIiEE.cl::sycl::specialization_id" { i32 42 }, align 4
@__builtin_unique_stable_name._ZN2cl4sycl14kernel_handler33getSpecializationConstantOnDeviceIL_ZL10SpecConst3E10TestStructLPv0EEET0_v = private unnamed_addr constant [75 x i8] c"_ZTSN2cl4sycl6detail32specialization_id_name_generatorIL_ZL10SpecConst3EEE\00", align 1
@_ZL10SpecConst3 = internal constant %"class._ZTSN2cl4sycl17specialization_idI10TestStructEE.cl::sycl::specialization_id" { %struct._ZTS10TestStruct.TestStruct { i32 42, i32 42 } }, align 4

; Function Attrs: convergent noinline norecurse
define weak_odr dso_local void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE11Kernel4Name"(%struct._ZTS10TestStruct.TestStruct addrspace(1)* %_arg_, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_1, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_2, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_3, i8 addrspace(1)* %_arg__specialization_constants_buffer) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !11 {
entry:
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = load i64, i64* %0, align 8
  %add.ptr.i = getelementptr inbounds %struct._ZTS10TestStruct.TestStruct, %struct._ZTS10TestStruct.TestStruct addrspace(1)* %_arg_, i64 %1
  %2 = addrspacecast i8 addrspace(1)* %_arg__specialization_constants_buffer to i8*
  %call.i.i.i = tail call %struct._ZTS10TestStruct.TestStruct @_Z40__sycl_getComposite2020SpecConstantValueI10TestStructET_PKcPKvS5_(i8* getelementptr inbounds ([75 x i8], [75 x i8]* @__builtin_unique_stable_name._ZN2cl4sycl14kernel_handler33getSpecializationConstantOnDeviceIL_ZL10SpecConst3E10TestStructLPv0EEET0_v, i64 0, i64 0), i8* bitcast (%"class._ZTSN2cl4sycl17specialization_idI10TestStructEE.cl::sycl::specialization_id"* @_ZL10SpecConst3 to i8*), i8* %2) #2
  %oldret.i.i.i = extractvalue %struct._ZTS10TestStruct.TestStruct %call.i.i.i, 0
  %oldret1.i.i.i = extractvalue %struct._ZTS10TestStruct.TestStruct %call.i.i.i, 1
  %ptridx.ascast.i.i = addrspacecast %struct._ZTS10TestStruct.TestStruct addrspace(1)* %add.ptr.i to %struct._ZTS10TestStruct.TestStruct*
  %ref.tmp.sroa.0.0..sroa_idx.i = getelementptr inbounds %struct._ZTS10TestStruct.TestStruct, %struct._ZTS10TestStruct.TestStruct* %ptridx.ascast.i.i, i64 0, i32 0
  store i32 %oldret.i.i.i, i32* %ref.tmp.sroa.0.0..sroa_idx.i, align 4, !tbaa.struct !12
  %ref.tmp.sroa.4.0..sroa_idx4.i = getelementptr inbounds %struct._ZTS10TestStruct.TestStruct, %struct._ZTS10TestStruct.TestStruct* %ptridx.ascast.i.i, i64 0, i32 1
  store i32 %oldret1.i.i.i, i32* %ref.tmp.sroa.4.0..sroa_idx4.i, align 4, !tbaa.struct !17
  ret void
}

; Function Attrs: convergent
declare dso_local %struct._ZTS10TestStruct.TestStruct @_Z40__sycl_getComposite2020SpecConstantValueI10TestStructET_PKcPKvS5_(i8*, i8*, i8*) local_unnamed_addr #1

attributes #0 = { convergent noinline norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="common.cpp" "target-cpu"="sm_50" "target-features"="+ptx64,+sm_50" "uniform-work-group-size"="true" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_50" "target-features"="+ptx64,+sm_50" }
attributes #2 = { convergent }

!llvm.module.flags = !{!0, !1}
!nvvm.annotations = !{!2, !3, !4, !3, !5, !5, !5, !5, !6, !6, !5}
!opencl.spir.version = !{!7}
!spirv.Source = !{!8}
!llvm.ident = !{!9}
!nvvmir.version = !{!10}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 1]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{void (%struct._ZTS10TestStruct.TestStruct addrspace(1)*, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"*, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"*, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"*, i8 addrspace(1)*)* @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE11Kernel4Name", !"kernel", i32 1}
!3 = !{null, !"align", i32 8}
!4 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!5 = !{null, !"align", i32 16}
!6 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!7 = !{i32 1, i32 2}
!8 = !{i32 4, i32 100000}
!9 = !{!"clang version 13.0.0"}
!10 = !{i32 1, i32 4}
!11 = !{i32 -1, i32 -1, i32 -1, i32 -1, i32 -1}
!12 = !{i64 0, i64 4, !13, i64 4, i64 4, !13}
!13 = !{!14, !14, i64 0}
!14 = !{!"int", !15, i64 0}
!15 = !{!"omnipotent char", !16, i64 0}
!16 = !{!"Simple C++ TBAA"}
!17 = !{i64 0, i64 4, !13}
