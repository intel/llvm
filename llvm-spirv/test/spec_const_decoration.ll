; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s

; CHECK: Decorate [[#SpecConst:]] SpecId 0
; CHECK: SpecConstant [[#]] [[#SpecConst]] 70
; CHECK: Phi [[#]] [[#]] [[#]] [[#]] [[#SpecConst]] [[#]]

source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }

$_ZTS6kernel = comdat any

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTS6kernel(i8 addrspace(1)* %_arg_, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_1, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_2, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_3) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !6 {
entry:
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range", %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast i64* %0 to i64 addrspace(4)*
  %2 = load i64, i64 addrspace(4)* %1, align 8
  br label %for.cond.i.i

for.cond.i.i:                                     ; preds = %for.body.i.i, %entry
  %value.0.i.i = phi i8 [ -1, %entry ], [ %3, %for.body.i.i ]
  %cmp.i.i = phi i1 [ true, %entry ], [ false, %for.body.i.i ]
  br i1 %cmp.i.i, label %for.body.i.i, label %_ZZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_ENKUlNS0_14kernel_handlerEE_clES4_.exit

for.body.i.i:                                     ; preds = %for.cond.i.i
  %3 = call i8 @_Z20__spirv_SpecConstantia(i32 0, i8 70)
  br label %for.cond.i.i, !llvm.loop !7

_ZZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_ENKUlNS0_14kernel_handlerEE_clES4_.exit: ; preds = %for.cond.i.i
  %add.ptr.i = getelementptr inbounds i8, i8 addrspace(1)* %_arg_, i64 %2
  %arrayidx.ascast.i.i = addrspacecast i8 addrspace(1)* %add.ptr.i to i8 addrspace(4)*
  store i8 %value.0.i.i, i8 addrspace(4)* %arrayidx.ascast.i.i, align 1, !tbaa !9
  ret void
}

declare i8 @_Z20__spirv_SpecConstantia(i32, i8)

attributes #0 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="s.cpp" "uniform-work-group-size"="true" }

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!llvm.ident = !{!2}
!llvm.module.flags = !{!3, !4}
!sycl.specialization-constants = !{!5}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 13.0.0 (/data/github.com/intel/llvm/clang 70d05d180d448c0dd03acf6aa3dbf9736a87bd46)"}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"_ZTSN2cl4sycl6detail32specialization_id_name_generatorIL_ZL10spec_constEEE", i32 0, i32 0, i32 1}
!6 = !{i32 -1, i32 -1, i32 -1, i32 -1}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.mustprogress"}
!9 = !{!10, !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
