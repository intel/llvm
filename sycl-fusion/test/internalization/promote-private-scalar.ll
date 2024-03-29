; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext\
; RUN: -passes=sycl-internalization -S %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

%0 = type { %1 }
%1 = type { [1 x i64] }

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 noundef %0) #1

; Function Attrs: nounwind willreturn
declare spir_func i64 @_Z13get_global_idj(i32 %0) #2

; Function Attrs: nounwind willreturn
declare spir_func i64 @_Z12get_group_idj(i32 %0) #2

; Function Attrs: nounwind willreturn
declare spir_func i64 @_Z20get_global_linear_idv() #2

; Function Attrs: nounwind willreturn
declare spir_func i64 @_Z14get_local_sizej(i32 %0) #2

; Function Attrs: alwaysinline nounwind
declare spir_func void @__itt_offload_wi_start_wrapper() #3

; Function Attrs: alwaysinline nounwind
declare spir_func void @__itt_offload_wi_finish_wrapper() #3

; Function Attrs: noinline nounwind
declare spir_func void @__itt_offload_wi_finish_stub(ptr addrspace(4) %group_id, i64 %wi_id) #4

; Function Attrs: noinline nounwind
declare spir_func void @__itt_offload_wi_start_stub(ptr addrspace(4) %group_id, i64 %wi_id, i32 %wg_size) #4


define spir_kernel void @fused_0(ptr addrspace(1) align 4 %KernelOne_accTmp, ptr byval(%0) align 8 %KernelOne_accTmp3, ptr addrspace(1) align 4 %KernelOne_accIn1, ptr byval(%0) align 8 %KernelOne_accIn16, ptr addrspace(1) align 4 %KernelOne_accIn2, ptr addrspace(1) align 4 %KernelTwo_accOut, ptr addrspace(1) align 4 %KernelTwo_accIn3) !kernel_arg_addr_space !12 !kernel_arg_access_qual !13 !kernel_arg_type !14 !kernel_arg_type_qual !15 !kernel_arg_base_type !14 !kernel_arg_name !16 !sycl.kernel.promote !17 !sycl.kernel.promote.localsize !18 !sycl.kernel.promote.elemsize !19 {
; Scenario: Test the successful private internalization of the first pointer
; argument. This means, the first pointer argument has been replaced by a 
; function-local alloca and all accesses have been updated to use this alloca
; instead.

; CHECK-LABEL: define {{[^@]+}}@fused_0
; CHECK-SAME: (ptr byval([[TYPE0:%.*]]) align 8 [[KERNELONE_ACCTMP3:%.*]], ptr addrspace(1) align 4 [[KERNELONE_ACCIN1:%.*]], ptr byval([[TYPE0]]) align 8 [[KERNELONE_ACCIN16:%.*]], ptr addrspace(1) align 4 [[KERNELONE_ACCIN2:%.*]], ptr addrspace(1) align 4 [[KERNELTWO_ACCOUT:%.*]], ptr addrspace(1) align 4 [[KERNELTWO_ACCIN3:%.*]])
; CHECK:  entry:
; CHECK:    [[TMP0:%.*]] = alloca i8, i64 4, align 4
; CHECK:    [[ADD_PTR_I_I:%.*]] = getelementptr inbounds float, ptr [[TMP0]], i64 0
; CHECK:    [[ADD_I_I:%.*]] = fadd float
; CHECK:    [[ARRAYIDX_I13_I_I:%.*]] = getelementptr inbounds float, ptr [[ADD_PTR_I_I]], i64 0
; CHECK:    store float [[ADD_I_I]], ptr [[ARRAYIDX_I13_I_I]], align 4
; CHECK:    [[ADD_PTR_I39_I8:%.*]] = getelementptr inbounds float, ptr [[TMP0]], i64 0
; CHECK:    [[ARRAYIDX_I_I_I11:%.*]] = getelementptr inbounds float, ptr [[ADD_PTR_I39_I8]], i64 0
; CHECK:    [[TMP16:%.*]] = load float, ptr [[ARRAYIDX_I_I_I11]], align 4
; CHECK:    [[MUL_I_I:%.*]] = fmul float [[TMP16]]
; CHECK:    store float [[MUL_I_I]]
; CHECK-NOT: store
; CHECK:    ret void
;
entry:
  %KernelOne_accIn163.sroa.0.0..sroa_idx = getelementptr inbounds %0, ptr %KernelOne_accIn16, i64 0, i32 0, i32 0, i64 0
  %KernelOne_accIn163.sroa.0.0.copyload = load i64, ptr %KernelOne_accIn163.sroa.0.0..sroa_idx, align 1
  %KernelOne_accIn162.sroa.0.0..sroa_idx = getelementptr inbounds %0, ptr %KernelOne_accIn16, i64 0, i32 0, i32 0, i64 0
  %KernelOne_accIn162.sroa.0.0.copyload = load i64, ptr %KernelOne_accIn162.sroa.0.0..sroa_idx, align 1
  %KernelOne_accTmp31.sroa.0.0..sroa_idx = getelementptr inbounds %0, ptr %KernelOne_accTmp3, i64 0, i32 0, i32 0, i64 0
  %KernelOne_accTmp31.sroa.0.0.copyload = load i64, ptr %KernelOne_accTmp31.sroa.0.0..sroa_idx, align 1
  call spir_func void @__itt_offload_wi_start_wrapper() #3
  %add.ptr.i.i = getelementptr inbounds float, ptr addrspace(1) %KernelOne_accTmp, i64 %KernelOne_accTmp31.sroa.0.0.copyload
  %add.ptr.i39.i = getelementptr inbounds float, ptr addrspace(1) %KernelOne_accIn1, i64 %KernelOne_accIn162.sroa.0.0.copyload
  %add.ptr.i53.i = getelementptr inbounds float, ptr addrspace(1) %KernelOne_accIn2, i64 %KernelOne_accIn163.sroa.0.0.copyload
  %0 = call spir_func i64 @_Z13get_global_idj(i32 0) #2
  %1 = insertelement <3 x i64> undef, i64 %0, i32 0
  %2 = call spir_func i64 @_Z13get_global_idj(i32 1) #2
  %3 = insertelement <3 x i64> %1, i64 %2, i32 1
  %4 = call spir_func i64 @_Z13get_global_idj(i32 2) #2
  %5 = insertelement <3 x i64> %3, i64 %4, i32 2
  %cmp.i.i.i = icmp ult i64 %0, 2147483648
  call void @llvm.assume(i1 %cmp.i.i.i)
  %arrayidx.i.i.i = getelementptr inbounds float, ptr addrspace(1) %add.ptr.i39.i, i64 %0
  %6 = load float, ptr addrspace(1) %arrayidx.i.i.i, align 4
  %arrayidx.i9.i.i = getelementptr inbounds float, ptr addrspace(1) %add.ptr.i53.i, i64 %0
  %7 = load float, ptr addrspace(1) %arrayidx.i9.i.i, align 4
  %add.i.i = fadd float %6, %7
  %arrayidx.i13.i.i = getelementptr inbounds float, ptr addrspace(1) %add.ptr.i.i, i64 %0
  store float %add.i.i, ptr addrspace(1) %arrayidx.i13.i.i, align 4
  call spir_func void @__itt_offload_wi_finish_wrapper() #3
  %KernelOne_accIn166.sroa.0.0..sroa_idx = getelementptr inbounds %0, ptr %KernelOne_accIn16, i64 0, i32 0, i32 0, i64 0
  %KernelOne_accIn166.sroa.0.0.copyload = load i64, ptr %KernelOne_accIn166.sroa.0.0..sroa_idx, align 1
  %KernelOne_accTmp35.sroa.0.0..sroa_idx = getelementptr inbounds %0, ptr %KernelOne_accTmp3, i64 0, i32 0, i32 0, i64 0
  %KernelOne_accTmp35.sroa.0.0.copyload = load i64, ptr %KernelOne_accTmp35.sroa.0.0..sroa_idx, align 1
  %KernelOne_accIn164.sroa.0.0..sroa_idx = getelementptr inbounds %0, ptr %KernelOne_accIn16, i64 0, i32 0, i32 0, i64 0
  %KernelOne_accIn164.sroa.0.0.copyload = load i64, ptr %KernelOne_accIn164.sroa.0.0..sroa_idx, align 1
  call spir_func void @__itt_offload_wi_start_wrapper() #3
  %add.ptr.i.i7 = getelementptr inbounds float, ptr addrspace(1) %KernelTwo_accOut, i64 %KernelOne_accIn164.sroa.0.0.copyload
  %add.ptr.i39.i8 = getelementptr inbounds float, ptr addrspace(1) %KernelOne_accTmp, i64 %KernelOne_accTmp35.sroa.0.0.copyload
  %add.ptr.i53.i9 = getelementptr inbounds float, ptr addrspace(1) %KernelTwo_accIn3, i64 %KernelOne_accIn166.sroa.0.0.copyload
  %8 = call spir_func i64 @_Z13get_global_idj(i32 0) #2
  %9 = insertelement <3 x i64> undef, i64 %8, i32 0
  %10 = call spir_func i64 @_Z13get_global_idj(i32 1) #2
  %11 = insertelement <3 x i64> %9, i64 %10, i32 1
  %12 = call spir_func i64 @_Z13get_global_idj(i32 2) #2
  %13 = insertelement <3 x i64> %11, i64 %12, i32 2
  %cmp.i.i.i10 = icmp ult i64 %8, 2147483648
  call void @llvm.assume(i1 %cmp.i.i.i10)
  %arrayidx.i.i.i11 = getelementptr inbounds float, ptr addrspace(1) %add.ptr.i39.i8, i64 %8
  %14 = load float, ptr addrspace(1) %arrayidx.i.i.i11, align 4
  %arrayidx.i9.i.i13 = getelementptr inbounds float, ptr addrspace(1) %add.ptr.i53.i9, i64 %8
  %15 = load float, ptr addrspace(1) %arrayidx.i9.i.i13, align 4
  %mul.i.i = fmul float %14, %15
  %arrayidx.i13.i.i15 = getelementptr inbounds float, ptr addrspace(1) %add.ptr.i.i7, i64 %8
  store float %mul.i.i, ptr addrspace(1) %arrayidx.i13.i.i15, align 4
  call spir_func void @__itt_offload_wi_finish_wrapper() #3
  ret void
}


attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #2 = { nounwind willreturn }
attributes #3 = { alwaysinline nounwind }
attributes #4 = { noinline nounwind }
attributes #5 = { nounwind }

!12 = !{i32 1, i32 0, i32 1, i32 0, i32 1, i32 1, i32 1}
!13 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!14 = !{!"ptr", !"class.sycl::_V1::range", !"ptr", !"class.sycl::_V1::range", !"ptr", !"ptr", !"ptr"}
!15 = !{!"", !"", !"", !"", !"", !"", !""}
!16 = !{!"KernelOne_accTmp", !"KernelOne_accTmp3", !"KernelOne_accIn1", !"KernelOne_accIn16", !"KernelOne_accIn2", !"KernelTwo_accOut", !"KernelTwo_accIn3"}
!17 = !{!"private", !"none", !"none", !"none", !"none", !"none", !"none"}
!18 = !{i64 1, !"", !"", !"", !"", !"", !""}
!19 = !{i64 4, !"", !"", !"", !"", !"", !""}
