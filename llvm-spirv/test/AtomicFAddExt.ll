; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_EXT_shader_atomic_float_add -o %t.spv
; RUN: llvm-spirv -to-text %t.spv -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown-sycldevice"

%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }
%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }

$_ZTSZZ3addIfEvvENKUlRN2cl4sycl7handlerEE19_14clES3_EUlNS1_4itemILi1ELb1EEEE23_37 = comdat any

$_ZTSZZ3addIdEvvENKUlRN2cl4sycl7handlerEE19_14clES3_EUlNS1_4itemILi1ELb1EEEE23_37 = comdat any

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; CHECK-SPIRV: Capability AtomicFloat32AddEXT
; CHECK-SPIRV: Capability AtomicFloat64AddEXT
; CHECK-SPIRV: Extension "SPV_EXT_shader_atomic_float_add"
; CHECK-SPIRV: TypeFloat [[TYPE_FLOAT_32:[0-9]+]] 32
; CHECK-SPIRV: TypeFloat [[TYPE_FLOAT_64:[0-9]+]] 64

; Function Attrs: convergent norecurse mustprogress
define weak_odr dso_local spir_kernel void @_ZTSZZ3addIfEvvENKUlRN2cl4sycl7handlerEE19_14clES3_EUlNS1_4itemILi1ELb1EEEE23_37(float addrspace(1)* %_arg_, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_1, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_2, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_3, float addrspace(1)* %_arg_4, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_6, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_7, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_8) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !4 {
entry:
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = load i64, i64* %0, align 8
  %add.ptr.i29 = getelementptr inbounds float, float addrspace(1)* %_arg_, i64 %1
  %2 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_8, i64 0, i32 0, i32 0, i64 0
  %3 = load i64, i64* %2, align 8
  %add.ptr.i = getelementptr inbounds float, float addrspace(1)* %_arg_4, i64 %3
  %4 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId to <3 x i64> addrspace(4)*), align 32, !noalias !5
  %5 = extractelement <3 x i64> %4, i64 0
  ; CHECK-SPIRV: 7 AtomicFAddEXT [[TYPE_FLOAT_32]]
  ; CHECK-LLVM: call spir_func float @[[FLOAT_FUNC_NAME:_Z21__spirv_AtomicFAddEXT[[:alnum:]]+]]({{.*}})
  %call3.i.i.i.i = tail call spir_func float @_Z21__spirv_AtomicFAddEXTPU3AS1fN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEf(float addrspace(1)* %add.ptr.i29, i32 1, i32 896, float 1.000000e+00) #2
  %add.i.i = fadd float %call3.i.i.i.i, 1.000000e+00
  %sext.i = shl i64 %5, 32
  %conv5.i = ashr exact i64 %sext.i, 32
  %ptridx.i.i = getelementptr inbounds float, float addrspace(1)* %add.ptr.i, i64 %conv5.i
  %ptridx.ascast.i.i = addrspacecast float addrspace(1)* %ptridx.i.i to float addrspace(4)*
  store float %add.i.i, float addrspace(4)* %ptridx.ascast.i.i, align 4, !tbaa !14
  ret void
}

; Function Attrs: convergent
; CHECK-LLVM: declare {{.*}}spir_func float @[[FLOAT_FUNC_NAME]](float addrspace(1)*, i32, i32, float)
declare dso_local spir_func float @_Z21__spirv_AtomicFAddEXTPU3AS1fN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEf(float addrspace(1)*, i32, i32, float) local_unnamed_addr #1

; Function Attrs: convergent norecurse mustprogress
define weak_odr dso_local spir_kernel void @_ZTSZZ3addIdEvvENKUlRN2cl4sycl7handlerEE19_14clES3_EUlNS1_4itemILi1ELb1EEEE23_37(double addrspace(1)* %_arg_, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_1, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_2, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_3, double addrspace(1)* %_arg_4, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_6, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_7, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_8) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !4 {
entry:
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = load i64, i64* %0, align 8
  %add.ptr.i29 = getelementptr inbounds double, double addrspace(1)* %_arg_, i64 %1
  %2 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_8, i64 0, i32 0, i32 0, i64 0
  %3 = load i64, i64* %2, align 8
  %add.ptr.i = getelementptr inbounds double, double addrspace(1)* %_arg_4, i64 %3
  %4 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId to <3 x i64> addrspace(4)*), align 32, !noalias !18
  %5 = extractelement <3 x i64> %4, i64 0
  ; CHECK-SPIRV: 7 AtomicFAddEXT [[TYPE_FLOAT_64]]
  ; CHECK-LLVM: call spir_func double @[[DOUBLE_FUNC_NAME:_Z21__spirv_AtomicFAddEXT[[:alnum:]]+]]({{.*}})
  %call3.i.i.i.i = tail call spir_func double @_Z21__spirv_AtomicFAddEXTPU3AS1dN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEd(double addrspace(1)* %add.ptr.i29, i32 1, i32 896, double 1.000000e+00) #2
  %add.i.i = fadd double %call3.i.i.i.i, 1.000000e+00
  %sext.i = shl i64 %5, 32
  %conv5.i = ashr exact i64 %sext.i, 32
  %ptridx.i.i = getelementptr inbounds double, double addrspace(1)* %add.ptr.i, i64 %conv5.i
  %ptridx.ascast.i.i = addrspacecast double addrspace(1)* %ptridx.i.i to double addrspace(4)*
  store double %add.i.i, double addrspace(4)* %ptridx.ascast.i.i, align 8, !tbaa !27
  ret void
}

; Function Attrs: convergent
; CHECK-LLVM: declare {{.*}}spir_func double @[[DOUBLE_FUNC_NAME]](double addrspace(1)*, i32, i32, double)
declare dso_local spir_func double @_Z21__spirv_AtomicFAddEXTPU3AS1dN5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagEd(double addrspace(1)*, i32, i32, double) local_unnamed_addr #1

attributes #0 = { convergent norecurse mustprogress "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="fadd.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 12.0.0"}
!4 = !{i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1}
!5 = !{!6, !8, !10, !12}
!6 = distinct !{!6, !7, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv: %agg.result"}
!7 = distinct !{!7, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv"}
!8 = distinct !{!8, !9, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v: %agg.result"}
!9 = distinct !{!9, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v"}
!10 = distinct !{!10, !11, !"_ZN2cl4sycl6detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!11 = distinct !{!11, !"_ZN2cl4sycl6detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!12 = distinct !{!12, !13, !"_ZN2cl4sycl6detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!13 = distinct !{!13, !"_ZN2cl4sycl6detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!14 = !{!15, !15, i64 0}
!15 = !{!"float", !16, i64 0}
!16 = !{!"omnipotent char", !17, i64 0}
!17 = !{!"Simple C++ TBAA"}
!18 = !{!19, !21, !23, !25}
!19 = distinct !{!19, !20, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv: %agg.result"}
!20 = distinct !{!20, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv"}
!21 = distinct !{!21, !22, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v: %agg.result"}
!22 = distinct !{!22, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v"}
!23 = distinct !{!23, !24, !"_ZN2cl4sycl6detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!24 = distinct !{!24, !"_ZN2cl4sycl6detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!25 = distinct !{!25, !26, !"_ZN2cl4sycl6detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!26 = distinct !{!26, !"_ZN2cl4sycl6detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!27 = !{!28, !28, i64 0}
!28 = !{!"double", !16, i64 0}
