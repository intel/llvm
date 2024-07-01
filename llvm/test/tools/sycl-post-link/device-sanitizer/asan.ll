; This test checks that the post-link tool properly generates "asanUsed=1"
; in [SYCL/misc properties]

; RUN: sycl-post-link -properties -split=kernel -symbols -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.prop
; CHECK: [SYCL/misc properties]
; CHECK: asanUsed=1

; ModuleID = 'parallel_for_int.cpp'
source_filename = "parallel_for_int.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E11MyKernelR_4 = comdat any

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__asan_func = internal addrspace(2) constant [106 x i8] c"typeinfo name for main::'lambda'(sycl::_V1::handler&)::operator()(sycl::_V1::handler&) const::MyKernelR_4\00"

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

; Function Attrs: mustprogress norecurse nounwind sanitize_address uwtable
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E11MyKernelR_4(ptr addrspace(1) noundef align 4 %_arg_array, i64 %__asan_launch) local_unnamed_addr #1 comdat !srcloc !7 !kernel_arg_buffer_location !8 !sycl_fixed_targets !9 {
entry:
  call spir_func void @__itt_offload_wi_start_wrapper()
  %0 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, align 32, !noalias !10
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %arrayidx.i = getelementptr inbounds i32, ptr addrspace(1) %_arg_array, i64 %0
  %1 = ptrtoint ptr addrspace(1) %arrayidx.i to i64
  call void @__asan_load4(i64 %1, i32 1, i64 %__asan_launch, ptr addrspace(2) null, i32 0, ptr addrspace(2) @__asan_func)
  %2 = load i32, ptr addrspace(1) %arrayidx.i, align 4, !tbaa !17
  %inc.i = add nsw i32 %2, 1
  store i32 %inc.i, ptr addrspace(1) %arrayidx.i, align 4, !tbaa !17
  call spir_func void @__itt_offload_wi_finish_wrapper()
  ret void
}

declare void @__asan_load4(i64, i32, i64, ptr addrspace(2), i32, ptr addrspace(2))

declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2), ...)

declare spir_func void @__itt_offload_wi_start_wrapper()

declare spir_func void @__itt_offload_wi_finish_wrapper()

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #1 = { mustprogress norecurse nounwind sanitize_address uwtable "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="parallel_for_int.cpp" "sycl-optlevel"="2" "uniform-work-group-size"="true" }

!llvm.module.flags = !{!0, !1, !2}
!opencl.spir.version = !{!3}
!spirv.Source = !{!4}
!llvm.ident = !{!5}
!device.sanitizer = !{!6}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 2}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 1, i32 2}
!4 = !{i32 4, i32 100000}
!5 = !{!"clang version 19.0.0git (https://github.com/intel/llvm f8eada76c08c6a5e6c5842842ac5b98fa72669be)"}
!6 = !{!"asan"}
!7 = !{i32 1536}
!8 = !{i32 -1, i32 -1}
!9 = !{}
!10 = !{!11, !13, !15}
!11 = distinct !{!11, !12, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!12 = distinct !{!12, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!13 = distinct !{!13, !14, !"_ZN7__spirv22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!14 = distinct !{!14, !"_ZN7__spirv22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!15 = distinct !{!15, !16, !"_ZNK4sycl3_V17nd_itemILi1EE13get_global_idEv: %agg.result"}
!16 = distinct !{!16, !"_ZNK4sycl3_V17nd_itemILi1EE13get_global_idEv"}
!17 = !{!18, !18, i64 0}
!18 = !{!"int", !19, i64 0}
!19 = !{!"omnipotent char", !20, i64 0}
!20 = !{!"Simple C++ TBAA"}
