;; The test serves a purpose to check if barrier instruction is being annotated
;; by SPIRITTAnnotations pass
;;
;; Compiled from https://github.com/intel/llvm-test-suite/blob/intel/SYCL/KernelAndProgram/kernel-and-program.cpp
;; with following commands:
;; clang++ -fsycl -fsycl-device-only kernel-and-program.cpp -o kernel_and_program_optimized.bc

; RUN: opt < %s --SPIRITTAnnotations -S | FileCheck %s
; RUN: opt < %s --SPIRITTAnnotations -enable-new-pm=1 -S | FileCheck %s

; ModuleID = 'kernel_and_program_optimized.bc'
source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown-sycldevice"

%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }

$_ZTSZ4mainE10SingleTask = comdat any

$_ZTSZ4mainE11ParallelFor = comdat any

$_ZTSZ4mainE13ParallelForND = comdat any

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInGlobalOffset = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: norecurse willreturn
define weak_odr dso_local spir_kernel void @_ZTSZ4mainE10SingleTask(i32 addrspace(1)* %_arg_, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_1, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_2, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_3) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !4 {
entry:
; CHECK-LABEL: _ZTSZ4mainE10SingleTask(
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @__itt_offload_wi_start_wrapper()
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range", %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast i64* %0 to i64 addrspace(4)*
  %2 = load i64, i64 addrspace(4)* %1, align 8
  %add.ptr.i = getelementptr inbounds i32, i32 addrspace(1)* %_arg_, i64 %2
  %ptridx.ascast.i9.i = addrspacecast i32 addrspace(1)* %add.ptr.i to i32 addrspace(4)*
  %3 = load i32, i32 addrspace(4)* %ptridx.ascast.i9.i, align 4, !tbaa !5
  %add.i = add nsw i32 %3, 1
  store i32 %add.i, i32 addrspace(4)* %ptridx.ascast.i9.i, align 4, !tbaa !5
; CHECK: call void @__itt_offload_wi_finish_wrapper()
; CHECK-NEXT: ret void
  ret void
}

; Function Attrs: norecurse willreturn
define weak_odr dso_local spir_kernel void @_ZTSZ4mainE11ParallelFor(i32 addrspace(1)* %_arg_, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_1, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_2, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_3) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !4 {
entry:
; CHECK-LABEL: _ZTSZ4mainE11ParallelFor(
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @__itt_offload_wi_start_wrapper()
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range", %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast i64* %0 to i64 addrspace(4)*
  %2 = load i64, i64 addrspace(4)* %1, align 8
  %add.ptr.i = getelementptr inbounds i32, i32 addrspace(1)* %_arg_, i64 %2
  %3 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId to <3 x i64> addrspace(4)*), align 32, !noalias !9
  %4 = extractelement <3 x i64> %3, i64 0
  %ptridx.i.i = getelementptr inbounds i32, i32 addrspace(1)* %add.ptr.i, i64 %4
  %ptridx.ascast.i.i = addrspacecast i32 addrspace(1)* %ptridx.i.i to i32 addrspace(4)*
  %5 = load i32, i32 addrspace(4)* %ptridx.ascast.i.i, align 4, !tbaa !5
  %add.i = add nsw i32 %5, 1
  store i32 %add.i, i32 addrspace(4)* %ptridx.ascast.i.i, align 4, !tbaa !5
; CHECK: call void @__itt_offload_wi_finish_wrapper()
; CHECK-NEXT: ret void
  ret void
}

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSZ4mainE13ParallelForND(i32 addrspace(3)* %_arg_, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_1, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_2, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_3, i32 addrspace(1)* %_arg_4, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_6, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_7, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_8) local_unnamed_addr #1 comdat !kernel_arg_buffer_location !16 {
entry:
; CHECK-LABEL: _ZTSZ4mainE13ParallelForND(
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @__itt_offload_wi_start_wrapper()
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range", %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* %_arg_8, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast i64* %0 to i64 addrspace(4)*
  %2 = load i64, i64 addrspace(4)* %1, align 8
  %add.ptr.i = getelementptr inbounds i32, i32 addrspace(1)* %_arg_4, i64 %2
  %3 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId to <3 x i64> addrspace(4)*), align 32, !noalias !17
  %4 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalOffset to <3 x i64> addrspace(4)*), align 32, !noalias !24
  %5 = extractelement <3 x i64> %3, i64 0
  %6 = extractelement <3 x i64> %4, i64 0
  %sub.i.i.i.i = sub i64 %5, %6
  %7 = trunc i64 %sub.i.i.i.i to i32
  %conv.i = and i32 %7, 1
  %xor.i = xor i32 %conv.i, 1
  %ptridx.i27.i = getelementptr inbounds i32, i32 addrspace(1)* %add.ptr.i, i64 %sub.i.i.i.i
  %ptridx.ascast.i28.i = addrspacecast i32 addrspace(1)* %ptridx.i27.i to i32 addrspace(4)*
  %8 = load i32, i32 addrspace(4)* %ptridx.ascast.i28.i, align 4, !tbaa !5
  %9 = zext i32 %conv.i to i64
  %ptridx.i23.i = getelementptr inbounds i32, i32 addrspace(3)* %_arg_, i64 %9
  %ptridx.ascast.i24.i = addrspacecast i32 addrspace(3)* %ptridx.i23.i to i32 addrspace(4)*
  store i32 %8, i32 addrspace(4)* %ptridx.ascast.i24.i, align 4, !tbaa !5
; CHECK: call void @__itt_offload_wg_barrier_wrapper()
; CHECK-NEXT: tail call void @_Z22__spirv_ControlBarrierjjj
; CHECK-NEXT: call void @__itt_offload_wi_resume_wrapper()
  tail call void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272) #3
  %conv6.i = zext i32 %xor.i to i64
  %ptridx.i17.i = getelementptr inbounds i32, i32 addrspace(3)* %_arg_, i64 %conv6.i
  %ptridx.ascast.i18.i = addrspacecast i32 addrspace(3)* %ptridx.i17.i to i32 addrspace(4)*
  %10 = load i32, i32 addrspace(4)* %ptridx.ascast.i18.i, align 4, !tbaa !5
  store i32 %10, i32 addrspace(4)* %ptridx.ascast.i28.i, align 4, !tbaa !5
; CHECK: call void @__itt_offload_wi_finish_wrapper()
; CHECK-NEXT: ret void
  ret void
}

; Function Attrs: convergent
declare dso_local void @_Z22__spirv_ControlBarrierjjj(i32, i32, i32) local_unnamed_addr #2

; CHECK: declare void @__itt_offload_wi_start_wrapper()
; CHECK: declare void @__itt_offload_wi_finish_wrapper()
; CHECK: declare void @__itt_offload_wg_barrier_wrapper()
; CHECK: declare void @__itt_offload_wi_resume_wrapper()

attributes #0 = { norecurse willreturn "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="llvm-test-suite/SYCL/KernelAndProgram/kernel-and-program.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent norecurse "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="llvm-test-suite/SYCL/KernelAndProgram/kernel-and-program.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { convergent }

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!llvm.ident = !{!2}
!llvm.module.flags = !{!3}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 13.0.0 (https://github.com/intel/llvm.git 3d2adc7b3ca269708bcabdc4a40352a5cacb4b9d)"}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 -1, i32 -1, i32 -1, i32 -1}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !12, !14}
!10 = distinct !{!10, !11, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv: %agg.result"}
!11 = distinct !{!11, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv"}
!12 = distinct !{!12, !13, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v: %agg.result"}
!13 = distinct !{!13, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v"}
!14 = distinct !{!14, !15, !"_ZN2cl4sycl6detail7Builder10getElementILi1EEEKNS0_2idIXT_EEEPS5_: %agg.result"}
!15 = distinct !{!15, !"_ZN2cl4sycl6detail7Builder10getElementILi1EEEKNS0_2idIXT_EEEPS5_"}
!16 = !{i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1}
!17 = !{!18, !20, !22}
!18 = distinct !{!18, !19, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv: %agg.result"}
!19 = distinct !{!19, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv"}
!20 = distinct !{!20, !21, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v: %agg.result"}
!21 = distinct !{!21, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v"}
!22 = distinct !{!22, !23, !"_ZN2cl4sycl6detail7Builder10getElementILi1EEEKNS0_7nd_itemIXT_EEEPS5_: %agg.result"}
!23 = distinct !{!23, !"_ZN2cl4sycl6detail7Builder10getElementILi1EEEKNS0_7nd_itemIXT_EEEPS5_"}
!24 = !{!25, !27, !22}
!25 = distinct !{!25, !26, !"_ZN7__spirv23InitSizesSTGlobalOffsetILi1EN2cl4sycl2idILi1EEEE8initSizeEv: %agg.result"}
!26 = distinct !{!26, !"_ZN7__spirv23InitSizesSTGlobalOffsetILi1EN2cl4sycl2idILi1EEEE8initSizeEv"}
!27 = distinct !{!27, !28, !"_ZN7__spirvL16initGlobalOffsetILi1EN2cl4sycl2idILi1EEEEET0_v: %agg.result"}
!28 = distinct !{!28, !"_ZN7__spirvL16initGlobalOffsetILi1EN2cl4sycl2idILi1EEEEET0_v"}
