; RUN: llvm-as < %s | llvm-spirv -spirv-text -o %t
; RUN: FileCheck < %t %s

; CHECK: Decorate [[#BOOL_CONST:]] SpecId [[#]]
; CHECK: TypeBool [[#BOOL_TY:]]
; CHECK: SpecConstantTrue [[#BOOL_TY]] [[#BOOL_CONST]]
; CHECK: Select [[#]] [[#]] [[#BOOL_CONST]]
; zext is also represented as Select because of how SPIR-V spec is written
; CHECK: Select [[#]] [[#]] [[#BOOL_CONST]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }
%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }

$"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel1" = comdat any

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel1"(i8 addrspace(1)* %_arg_, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_1, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_2, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_3) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !4 {
entry:
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast i64* %0 to i64 addrspace(4)*
  %2 = load i64, i64 addrspace(4)* %1, align 8
  %add.ptr.i = getelementptr inbounds i8, i8 addrspace(1)* %_arg_, i64 %2
  %3 = call i1 @_Z20__spirv_SpecConstantia(i32 0, i8 1), !SYCL_SPEC_CONST_SYM_ID !5
  %ptridx.ascast.i.i = addrspacecast i8 addrspace(1)* %add.ptr.i to i8 addrspace(4)*
  %selected = select i1 %3, i8 0, i8 1
  %frombool.i = zext i1 %3 to i8
  %sum = add i8 %frombool.i, %selected
  store i8 %selected, i8 addrspace(4)* %ptridx.ascast.i.i, align 1, !tbaa !6
  ret void
}

declare i1 @_Z20__spirv_SpecConstantia(i32, i8)

attributes #0 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="t.cpp" "uniform-work-group-size"="true" }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 13.0.0 (/data/github.com/intel/llvm/clang c1f0fd875de45645c66776667682af55059968b7)"}
!4 = !{i32 -1, i32 -1, i32 -1, i32 -1}
!5 = !{!"_ZTSN2cl4sycl6detail32specialization_id_name_generatorIL_ZL7bool_idEEE", i32 0}
!6 = !{!7, !7, i64 0}
!7 = !{!"bool", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
