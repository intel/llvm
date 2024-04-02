;; The test serves a purpose to check if Atomic store instruction is being
;; annotated by SPIRITTAnnotations pass
;;
;; Compiled from https://github.com/intel/llvm/blob/sycl/sycl/test-e2e/AtomicRef/load.cpp
;; with following commands:
;; clang++ -fsycl -fsycl-device-only load.cpp -o load.bc

; RUN: opt < %s -passes=SPIRITTAnnotations -S | FileCheck %s
; RUN: opt < %s -passes=SPIRITTAnnotations -S | FileCheck %s

; ModuleID = 'store.bc'
source_filename = "llvm-test-suite/SYCL/AtomicRef/store.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }
%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }

$_ZTSN2cl4sycl6detail19__pf_kernel_wrapperI12store_kernelIiEEE = comdat any

$_ZTS12store_kernelIiE = comdat any

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSN2cl4sycl6detail19__pf_kernel_wrapperI12store_kernelIiEEE(ptr byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_, ptr addrspace(1) %_arg_1, ptr byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_2, ptr byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_3, ptr byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_4) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !4 {
entry:
; CHECK-LABEL: _ZTSN2cl4sycl6detail19__pf_kernel_wrapperI12store_kernelIiEEE(
; CHECK-NEXT: entry:
; CHECK-NEXT: call spir_func void @__itt_offload_wi_start_wrapper()
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range", ptr %_arg_, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = load i64, ptr addrspace(4) %1, align 8
  %3 = load <3 x i64>, ptr  addrspace(4) addrspacecast (ptr  addrspace(1) @__spirv_BuiltInGlobalInvocationId to ptr  addrspace(4)), align 32, !noalias !5
  %4 = extractelement <3 x i64> %3, i64 0
  %cmp.not.i = icmp ult i64 %4, %2
  br i1 %cmp.not.i, label %if.end.i, label %_ZZN2cl4sycl7handler24parallel_for_lambda_implI12store_kernelIiEZZ10store_testIiEvNS0_5queueEmENKUlRS1_E_clES7_EUlNS0_4itemILi1ELb1EEEE_Li1EEEvNS0_5rangeIXT1_EEET0_ENKUlSA_E_clESA_.exit

if.end.i:                                         ; preds = %entry
  %5 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", ptr %_arg_4, i64 0, i32 0, i32 0, i64 0
  %6 = addrspacecast ptr %5 to ptr addrspace(4)
  %7 = load i64, ptr addrspace(4) %6, align 8
  %add.ptr.i = getelementptr inbounds i32, ptr addrspace(1) %_arg_1, i64 %7
  %conv.i.i = trunc i64 %4 to i32
; CHECK: [[ARG_ASCAST:%[0-9a-zA-Z._]+]] = addrspacecast ptr addrspace(1) %[[ATOMIC_ARG_1:[0-9a-zA-Z._]+]] to ptr addrspace(4)
; CHECK-NEXT: call spir_func void @__itt_offload_atomic_op_start(ptr addrspace(4) [[ARG_ASCAST]], i32 1, i32 0
; CHECK-NEXT: {{.*}}__spirv_AtomicStore{{.*}}(ptr addrspace(1) %[[ATOMIC_ARG_1]],{{.*}}, i32 896
; CHECK-NEXT: [[ARG_ASCAST:%[0-9a-zA-Z._]+]] = addrspacecast ptr addrspace(1) %[[ATOMIC_ARG_1]] to ptr addrspace(4)
; CHECK-NEXT: call spir_func void @__itt_offload_atomic_op_finish(ptr addrspace(4) [[ARG_ASCAST]], i32 1, i32 0
  tail call spir_func void @_Z19__spirv_AtomicStorePU3AS1iN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEi(ptr addrspace(1) %add.ptr.i, i32 1, i32 896, i32 %conv.i.i) #2
  tail call spir_func void @__synthetic_spir_fun_call(ptr addrspace(1) %add.ptr.i)
  br label %_ZZN2cl4sycl7handler24parallel_for_lambda_implI12store_kernelIiEZZ10store_testIiEvNS0_5queueEmENKUlRS1_E_clES7_EUlNS0_4itemILi1ELb1EEEE_Li1EEEvNS0_5rangeIXT1_EEET0_ENKUlSA_E_clESA_.exit

_ZZN2cl4sycl7handler24parallel_for_lambda_implI12store_kernelIiEZZ10store_testIiEvNS0_5queueEmENKUlRS1_E_clES7_EUlNS0_4itemILi1ELb1EEEE_Li1EEEvNS0_5rangeIXT1_EEET0_ENKUlSA_E_clESA_.exit: ; preds = %entry, %if.end.i
; CHECK: call spir_func void @__itt_offload_wi_finish_wrapper()
; CHECK-NEXT: ret void
  ret void
}

define weak_odr dso_local spir_func void @__synthetic_spir_fun_call(ptr addrspace(1) %ptr) {
entry:
; CHECK-LABEL: spir_func void @__synthetic_spir_fun_call(ptr addrspace(1) %ptr) {
; CHECK: [[ARG_ASCAST:%[0-9a-zA-Z._]+]] = addrspacecast ptr addrspace(1) %[[ATOMIC_ARG_S:[0-9a-zA-Z._]+]] to ptr addrspace(4)
; CHECK-NEXT: call spir_func void @__itt_offload_atomic_op_start(ptr addrspace(4) [[ARG_ASCAST]], i32 1, i32 0)
; CHECK-NEXT: {{.*}}__spirv_AtomicStore{{.*}}(ptr addrspace(1) %[[ATOMIC_ARG_S]],{{.*}}, i32 896
; CHECK-NEXT: [[ARG_ASCAST:%[0-9a-zA-Z._]+]] = addrspacecast ptr addrspace(1) %[[ATOMIC_ARG_S]] to ptr addrspace(4)
; CHECK-NEXT: call spir_func void @__itt_offload_atomic_op_finish(ptr addrspace(4) [[ARG_ASCAST]], i32 1, i32 0)
  %0 = load <3 x i64>, ptr  addrspace(4) addrspacecast (ptr  addrspace(1) @__spirv_BuiltInGlobalInvocationId to ptr  addrspace(4)), align 32, !noalias !15
  %1 = extractelement <3 x i64> %0, i64 0
  %conv = trunc i64 %1 to i32
  tail call spir_func void @_Z19__spirv_AtomicStorePU3AS1iN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEi(ptr addrspace(1) %ptr, i32 1, i32 896, i32 %conv) #2
; CHECK-NOT: call spir_func void @__itt_offload_wi_finish_wrapper()
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func void @_Z19__spirv_AtomicStorePU3AS1iN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEi(ptr addrspace(1), i32, i32, i32) local_unnamed_addr #1

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTS12store_kernelIiE(ptr addrspace(1) %_arg_, ptr byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_1, ptr byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_2, ptr byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_3) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !14 {
entry:
; CHECK-LABEL: _ZTS12store_kernelIiE(
; CHECK-NEXT: entry:
; CHECK-NEXT: call spir_func void @__itt_offload_wi_start_wrapper()
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", ptr %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = load i64, ptr addrspace(4) %1, align 8
  %add.ptr.i = getelementptr inbounds i32, ptr addrspace(1) %_arg_, i64 %2
  %3 = load <3 x i64>, ptr  addrspace(4) addrspacecast (ptr  addrspace(1) @__spirv_BuiltInGlobalInvocationId to ptr  addrspace(4)), align 32, !noalias !15
  %4 = extractelement <3 x i64> %3, i64 0
  %conv.i = trunc i64 %4 to i32
; CHECK: [[ARG_ASCAST:%[0-9a-zA-Z._]+]] = addrspacecast ptr addrspace(1) %[[ATOMIC_ARG_2:[0-9a-zA-Z._]+]] to ptr addrspace(4)
; CHECK-NEXT: call spir_func void @__itt_offload_atomic_op_start(ptr addrspace(4) [[ARG_ASCAST]], i32 1, i32 0)
; CHECK-NEXT: {{.*}}__spirv_AtomicStore{{.*}}(ptr addrspace(1) %[[ATOMIC_ARG_2]],{{.*}}, i32 896
; CHECK-NEXT: [[ARG_ASCAST:%[0-9a-zA-Z._]+]] = addrspacecast ptr addrspace(1) %[[ATOMIC_ARG_2]] to ptr addrspace(4)
; CHECK-NEXT: call spir_func void @__itt_offload_atomic_op_finish(ptr addrspace(4) [[ARG_ASCAST]], i32 1, i32 0)
  tail call spir_func void @_Z19__spirv_AtomicStorePU3AS1iN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEi(ptr addrspace(1) %add.ptr.i, i32 1, i32 896, i32 %conv.i) #2
; CHECK: call spir_func void @__itt_offload_wi_finish_wrapper()
; CHECK-NEXT: ret void
  ret void
}

; CHECK: declare spir_func void @__itt_offload_wi_start_wrapper()
; CHECK: declare spir_func void @__itt_offload_atomic_op_start(ptr addrspace(4), i32, i32)
; CHECK: declare spir_func void @__itt_offload_atomic_op_finish(ptr addrspace(4), i32, i32)
; CHECK: declare spir_func void @__itt_offload_wi_finish_wrapper()

attributes #0 = { convergent norecurse "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="llvm-test-suite/SYCL/AtomicRef/store.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 13.0.0 (https://github.com/intel/llvm.git 51f22c4b69cf01465bdd7b586343f6e19e9ab045)"}
!4 = !{i32 -1, i32 -1, i32 -1, i32 -1, i32 -1}
!5 = !{!6, !8, !10, !12}
!6 = distinct !{!6, !7, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv: %agg.result"}
!7 = distinct !{!7, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv"}
!8 = distinct !{!8, !9, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v: %agg.result"}
!9 = distinct !{!9, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v"}
!10 = distinct !{!10, !11, !"_ZN2cl4sycl6detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!11 = distinct !{!11, !"_ZN2cl4sycl6detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!12 = distinct !{!12, !13, !"_ZN2cl4sycl6detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!13 = distinct !{!13, !"_ZN2cl4sycl6detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!14 = !{i32 -1, i32 -1, i32 -1, i32 -1}
!15 = !{!16, !18, !20, !22}
!16 = distinct !{!16, !17, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv: %agg.result"}
!17 = distinct !{!17, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv"}
!18 = distinct !{!18, !19, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v: %agg.result"}
!19 = distinct !{!19, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v"}
!20 = distinct !{!20, !21, !"_ZN2cl4sycl6detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!21 = distinct !{!21, !"_ZN2cl4sycl6detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!22 = distinct !{!22, !23, !"_ZN2cl4sycl6detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!23 = distinct !{!23, !"_ZN2cl4sycl6detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
