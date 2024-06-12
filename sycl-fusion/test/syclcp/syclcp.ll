; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext\
; RUN: -passes=sycl-cp -S %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

%0 = type { %1 }
%1 = type { [1 x i64] }
%2 = type { [10 x i32] }
%3 = type { %4, %4, [10 x i32] }
%4 = type { %5, %6 }
%5 = type { %0, %0, %0 }
%6 = type { ptr addrspace(1) }

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0i8(i64 immarg %0, ptr nocapture %1) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0i8(i64 immarg %0, ptr nocapture %1) #0

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

; Function Attrs: nounwind willreturn
declare spir_func i64 @_Z12get_local_idj(i32 %0) #2

; Function Attrs: nounwind willreturn
declare spir_func i64 @_Z17get_global_offsetj(i32 %0) #2

; Function Attrs: alwaysinline nounwind
declare spir_func void @__itt_offload_wi_start_wrapper() #4

; Function Attrs: alwaysinline nounwind
declare spir_func void @__itt_offload_wi_finish_wrapper() #4

; Function Attrs: noinline nounwind
declare spir_func void @__itt_offload_wi_finish_stub(ptr addrspace(4) %group_id, i64 %wi_id) #5

; Function Attrs: noinline nounwind
declare spir_func void @__itt_offload_wi_start_stub(ptr addrspace(4) %group_id, i64 %wi_id, i32 %wg_size) #5


define spir_kernel void @fused_0(ptr byval(%0) align 8 %KernelOne_accTmp3, ptr addrspace(1) align 4 %KernelOne_accIn1, ptr byval(%0) align 8 %KernelOne_accIn16, ptr addrspace(1) align 4 %KernelOne_accIn2, ptr addrspace(1) align 4 %KernelTwo_out, ptr byval(%2) align 4 %KernelTwo_coef) !kernel_arg_addr_space !16 !kernel_arg_access_qual !7 !kernel_arg_type !17 !kernel_arg_type_qual !9 !kernel_arg_base_type !17 !kernel_arg_name !18 !sycl.kernel.constants !19 {
; Scenario: Test constant propagation. Propagates a scalar ([1x i64]) and
; an aggregate ([10xi32]) constant. The test mainly verifies that the function
; signature has been updated, i.e., the propagated arguments have been removed,
; and the initialization uses the correct values.

; CHECK: [[TYPE2:%.*]] = type { [10 x i32] }
; CHECK: define {{[^@]+}}@fused_0
; CHECK-SAME: (ptr byval([[TYPE0:%.*]]) align 8 [[KernelOne_ACCTMP3:%.*]], ptr addrspace(1) align 4 [[KernelOne_ACCIN1:%.*]], ptr addrspace(1) align 4 [[KernelOne_ACCIN2:%.*]], ptr addrspace(1) align 4 [[KernelTwo_OUT:%.*]])
; CHECK:  entry:
; CHECK:    [[TMP0:%.*]] = alloca [[TYPE0]], align 8
; CHECK:    [[TMP1:%.*]] = getelementptr inbounds [[TYPE0]], ptr [[TMP0]], i32 0, i32 0, i32 0, i32 0
; CHECK:    store i64 0, ptr [[TMP1]], align 8
; CHECK:    [[TMP2:%.*]] = alloca [[TYPE2]], align 8
; CHECK:    [[TMP3:%.*]] = getelementptr inbounds [[TYPE2]], ptr [[TMP2]], i32 0, i32 0, i32 0
; CHECK:    store i32 0, ptr [[TMP3]], align 4
; CHECK:    [[TMP4:%.*]] = getelementptr inbounds [[TYPE2]], ptr [[TMP2]], i32 0, i32 0, i32 1
; CHECK:    store i32 1, ptr [[TMP4]], align 4
; CHECK:    [[TMP5:%.*]] = getelementptr inbounds [[TYPE2]], ptr [[TMP2]], i32 0, i32 0, i32 2
; CHECK:    store i32 2, ptr [[TMP5]], align 4
; CHECK:    [[TMP6:%.*]] = getelementptr inbounds [[TYPE2]], ptr [[TMP2]], i32 0, i32 0, i32 3
; CHECK:    store i32 3, ptr [[TMP6]], align 4
; CHECK:    [[TMP7:%.*]] = getelementptr inbounds [[TYPE2]], ptr [[TMP2]], i32 0, i32 0, i32 4
; CHECK:    store i32 4, ptr [[TMP7]], align 4
; CHECK:    [[TMP8:%.*]] = getelementptr inbounds [[TYPE2]], ptr [[TMP2]], i32 0, i32 0, i32 5
; CHECK:    store i32 5, ptr [[TMP8]], align 4
; CHECK:    [[TMP9:%.*]] = getelementptr inbounds [[TYPE2]], ptr [[TMP2]], i32 0, i32 0, i32 6
; CHECK:    store i32 6, ptr [[TMP9]], align 4
; CHECK:    [[TMP10:%.*]] = getelementptr inbounds [[TYPE2]], ptr [[TMP2]], i32 0, i32 0, i32 7
; CHECK:    store i32 7, ptr [[TMP10]], align 4
; CHECK:    [[TMP11:%.*]] = getelementptr inbounds [[TYPE2]], ptr [[TMP2]], i32 0, i32 0, i32 8
; CHECK:    store i32 8, ptr [[TMP11]], align 4
; CHECK:    [[TMP12:%.*]] = getelementptr inbounds [[TYPE2]], ptr [[TMP2]], i32 0, i32 0, i32 9
; CHECK:    store i32 9, ptr [[TMP12]], align 4
; CHECK:    [[TMP13:%.*]] = alloca [1 x i32], align 4
; CHECK:    [[TMP14:%.*]] = getelementptr inbounds [1 x i32], ptr [[TMP13]], i64 0, i64 0
; CHECK:    [[KERNELTWO_I:%.*]] = alloca [[TMP3]], align 8
; CHECK:    [[KernelOne_ACCIN163_SROA_0_0__SROA_IDX:%.*]] = getelementptr inbounds [[TMP0]], ptr [[TMP0]], i64 0, i32 0, i32 0, i64 0
; CHECK:    [[KernelOne_ACCIN163_SROA_0_0_COPYLOAD:%.*]] = load i64, ptr
; CHECK:    [[KernelTwo_COEF6_SROA_0_0__SROA_IDX:%.*]] = getelementptr inbounds [[TYPE2]], ptr [[TMP2]], i64 0, i32 0, i64 0
; CHECK:    [[KernelTwo_COEF6_SROA_0_0_COPYLOAD:%.*]] = load i32, ptr [[KernelTwo_COEF6_SROA_0_0__SROA_IDX]], align 1
; CHECK:    ret void
;
entry:
  %0 = alloca [1 x i32], align 4
  %1 = getelementptr inbounds [1 x i32], ptr %0, i64 0, i64 0
  %KernelTwo.i = alloca %3, align 8
  %KernelOne_accIn163.sroa.0.0..sroa_idx = getelementptr inbounds %0, ptr %KernelOne_accIn16, i64 0, i32 0, i32 0, i64 0
  %KernelOne_accIn163.sroa.0.0.copyload = load i64, ptr %KernelOne_accIn163.sroa.0.0..sroa_idx, align 1
  %KernelOne_accIn162.sroa.0.0..sroa_idx = getelementptr inbounds %0, ptr %KernelOne_accIn16, i64 0, i32 0, i32 0, i64 0
  %KernelOne_accIn162.sroa.0.0.copyload = load i64, ptr %KernelOne_accIn162.sroa.0.0..sroa_idx, align 1
  %KernelOne_accTmp31.sroa.0.0..sroa_idx = getelementptr inbounds %0, ptr %KernelOne_accTmp3, i64 0, i32 0, i32 0, i64 0
  %KernelOne_accTmp31.sroa.0.0.copyload = load i64, ptr %KernelOne_accTmp31.sroa.0.0..sroa_idx, align 1
  call spir_func void @__itt_offload_wi_start_wrapper() #4
  %add.ptr.i.i = getelementptr inbounds i32, ptr %1, i64 0
  %add.ptr.i39.i = getelementptr inbounds i32, ptr addrspace(1) %KernelOne_accIn1, i64 %KernelOne_accIn162.sroa.0.0.copyload
  %add.ptr.i53.i = getelementptr inbounds i32, ptr addrspace(1) %KernelOne_accIn2, i64 %KernelOne_accIn163.sroa.0.0.copyload
  %2 = call spir_func i64 @_Z13get_global_idj(i32 0) #2
  %3 = call spir_func i64 @_Z13get_global_idj(i32 1) #2
  %4 = call spir_func i64 @_Z13get_global_idj(i32 2) #2
  %cmp.i.i.i = icmp ult i64 %2, 2147483648
  call void @llvm.assume(i1 %cmp.i.i.i)
  %arrayidx.i.i.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i39.i, i64 %2
  %5 = load i32, ptr addrspace(1) %arrayidx.i.i.i, align 4
  %arrayidx.i9.i.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i53.i, i64 %2
  %6 = load i32, ptr addrspace(1) %arrayidx.i9.i.i, align 4
  %add.i.i = add nsw i32 %5, %6
  %arrayidx.i13.i.i = getelementptr inbounds i32, ptr %add.ptr.i.i, i64 0
  store i32 %add.i.i, ptr %arrayidx.i13.i.i, align 4
  call spir_func void @__itt_offload_wi_finish_wrapper() #4
  %KernelTwo_coef6.sroa.0.0..sroa_idx = getelementptr inbounds %2, ptr %KernelTwo_coef, i64 0, i32 0, i64 0
  %KernelTwo_coef6.sroa.0.0.copyload = load i32, ptr %KernelTwo_coef6.sroa.0.0..sroa_idx, align 1
  %KernelTwo_coef6.sroa.4.0..sroa_idx17 = getelementptr inbounds %2, ptr %KernelTwo_coef, i64 0, i32 0, i64 1
  %KernelTwo_coef6.sroa.4.0.copyload = load i32, ptr %KernelTwo_coef6.sroa.4.0..sroa_idx17, align 1
  %KernelTwo_coef6.sroa.5.0..sroa_idx19 = getelementptr inbounds %2, ptr %KernelTwo_coef, i64 0, i32 0, i64 2
  %KernelTwo_coef6.sroa.5.0.copyload = load i32, ptr %KernelTwo_coef6.sroa.5.0..sroa_idx19, align 1
  %KernelTwo_coef6.sroa.6.0..sroa_idx21 = getelementptr inbounds %2, ptr %KernelTwo_coef, i64 0, i32 0, i64 3
  %KernelTwo_coef6.sroa.6.0.copyload = load i32, ptr %KernelTwo_coef6.sroa.6.0..sroa_idx21, align 1
  %KernelTwo_coef6.sroa.7.0..sroa_idx23 = getelementptr inbounds %2, ptr %KernelTwo_coef, i64 0, i32 0, i64 4
  %KernelTwo_coef6.sroa.7.0.copyload = load i32, ptr %KernelTwo_coef6.sroa.7.0..sroa_idx23, align 1
  %KernelTwo_coef6.sroa.8.0..sroa_idx25 = getelementptr inbounds %2, ptr %KernelTwo_coef, i64 0, i32 0, i64 5
  %KernelTwo_coef6.sroa.8.0.copyload = load i32, ptr %KernelTwo_coef6.sroa.8.0..sroa_idx25, align 1
  %KernelTwo_coef6.sroa.9.0..sroa_idx27 = getelementptr inbounds %2, ptr %KernelTwo_coef, i64 0, i32 0, i64 6
  %KernelTwo_coef6.sroa.9.0.copyload = load i32, ptr %KernelTwo_coef6.sroa.9.0..sroa_idx27, align 1
  %KernelTwo_coef6.sroa.10.0..sroa_idx29 = getelementptr inbounds %2, ptr %KernelTwo_coef, i64 0, i32 0, i64 7
  %KernelTwo_coef6.sroa.10.0.copyload = load i32, ptr %KernelTwo_coef6.sroa.10.0..sroa_idx29, align 1
  %KernelTwo_coef6.sroa.11.0..sroa_idx31 = getelementptr inbounds %2, ptr %KernelTwo_coef, i64 0, i32 0, i64 8
  %KernelTwo_coef6.sroa.11.0.copyload = load i32, ptr %KernelTwo_coef6.sroa.11.0..sroa_idx31, align 1
  %KernelTwo_coef6.sroa.12.0..sroa_idx33 = getelementptr inbounds %2, ptr %KernelTwo_coef, i64 0, i32 0, i64 9
  %KernelTwo_coef6.sroa.12.0.copyload = load i32, ptr %KernelTwo_coef6.sroa.12.0..sroa_idx33, align 1
  %KernelOne_accIn165.sroa.0.0..sroa_idx = getelementptr inbounds %0, ptr %KernelOne_accIn16, i64 0, i32 0, i32 0, i64 0
  %KernelOne_accIn165.sroa.0.0.copyload = load i64, ptr %KernelOne_accIn165.sroa.0.0..sroa_idx, align 1
  %KernelOne_accTmp34.sroa.0.0..sroa_idx = getelementptr inbounds %0, ptr %KernelOne_accTmp3, i64 0, i32 0, i32 0, i64 0
  %KernelOne_accTmp34.sroa.0.0.copyload = load i64, ptr %KernelOne_accTmp34.sroa.0.0..sroa_idx, align 1
  call spir_func void @__itt_offload_wi_start_wrapper() #4
  %7 = bitcast ptr %KernelTwo.i to ptr
  call void @llvm.lifetime.start.p0i8(i64 104, ptr %7)
  br label %arrayinit.body.i

arrayinit.body.i:                                 ; preds = %entry
  %8 = getelementptr inbounds %3, ptr %KernelTwo.i, i64 0, i32 2, i64 0
  store i32 %KernelTwo_coef6.sroa.0.0.copyload, ptr %8, align 4
  %9 = getelementptr inbounds %3, ptr %KernelTwo.i, i64 0, i32 2, i64 1
  store i32 %KernelTwo_coef6.sroa.4.0.copyload, ptr %9, align 4
  %10 = getelementptr inbounds %3, ptr %KernelTwo.i, i64 0, i32 2, i64 2
  store i32 %KernelTwo_coef6.sroa.5.0.copyload, ptr %10, align 4
  %11 = getelementptr inbounds %3, ptr %KernelTwo.i, i64 0, i32 2, i64 3
  store i32 %KernelTwo_coef6.sroa.6.0.copyload, ptr %11, align 4
  %12 = getelementptr inbounds %3, ptr %KernelTwo.i, i64 0, i32 2, i64 4
  store i32 %KernelTwo_coef6.sroa.7.0.copyload, ptr %12, align 4
  %13 = getelementptr inbounds %3, ptr %KernelTwo.i, i64 0, i32 2, i64 5
  store i32 %KernelTwo_coef6.sroa.8.0.copyload, ptr %13, align 4
  %14 = getelementptr inbounds %3, ptr %KernelTwo.i, i64 0, i32 2, i64 6
  store i32 %KernelTwo_coef6.sroa.9.0.copyload, ptr %14, align 4
  %15 = getelementptr inbounds %3, ptr %KernelTwo.i, i64 0, i32 2, i64 7
  store i32 %KernelTwo_coef6.sroa.10.0.copyload, ptr %15, align 4
  %16 = getelementptr inbounds %3, ptr %KernelTwo.i, i64 0, i32 2, i64 8
  store i32 %KernelTwo_coef6.sroa.11.0.copyload, ptr %16, align 4
  %17 = getelementptr inbounds %3, ptr %KernelTwo.i, i64 0, i32 2, i64 9
  store i32 %KernelTwo_coef6.sroa.12.0.copyload, ptr %17, align 4
  %add.ptr.i.i7 = getelementptr inbounds i32, ptr %1, i64 0
  %add.ptr.i33.i = getelementptr inbounds i32, ptr addrspace(1) %KernelTwo_out, i64 %KernelOne_accIn165.sroa.0.0.copyload
  %18 = call spir_func i64 @_Z13get_global_idj(i32 0) #2
  %19 = call spir_func i64 @_Z13get_global_idj(i32 1) #2
  %20 = call spir_func i64 @_Z13get_global_idj(i32 2) #2
  %21 = call spir_func i64 @_Z12get_local_idj(i32 0) #2
  %22 = call spir_func i64 @_Z12get_local_idj(i32 1) #2
  %23 = call spir_func i64 @_Z12get_local_idj(i32 2) #2
  %24 = call spir_func i64 @_Z17get_global_offsetj(i32 0) #2
  %25 = call spir_func i64 @_Z17get_global_offsetj(i32 1) #2
  %26 = call spir_func i64 @_Z17get_global_offsetj(i32 2) #2
  %sub.i.i.i.i.i = sub i64 %18, %24
  %cmp.i.i.i8 = icmp ult i64 %sub.i.i.i.i.i, 2147483648
  call void @llvm.assume(i1 %cmp.i.i.i8)
  %arrayidx.i.i.i9 = getelementptr inbounds i32, ptr %add.ptr.i.i7, i64 0
  %27 = load i32, ptr %arrayidx.i.i.i9, align 4
  %cmp.i13.i.i = icmp ult i64 %21, 2147483648
  call void @llvm.assume(i1 %cmp.i13.i.i)
  %arrayidx.i.i = getelementptr inbounds %3, ptr %KernelTwo.i, i64 0, i32 2, i64 %21
  %28 = load i32, ptr %arrayidx.i.i, align 4
  %mul.i.i = mul nsw i32 %27, %28
  %arrayidx.i17.i.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i33.i, i64 %sub.i.i.i.i.i
  store i32 %mul.i.i, ptr addrspace(1) %arrayidx.i17.i.i, align 4
  call void @llvm.lifetime.end.p0i8(i64 104, ptr %7)
  call spir_func void @__itt_offload_wi_finish_wrapper() #4
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #2 = { nounwind willreturn }
attributes #3 = { nounwind }
attributes #4 = { alwaysinline nounwind }
attributes #5 = { noinline nounwind }

!7 = !{!"none", !"none", !"none", !"none", !"none", !"none"}
!9 = !{!"", !"", !"", !"", !"", !""}
!16 = !{i32 0, i32 1, i32 0, i32 1, i32 1, i32 0}
!17 = !{!"class.sycl::_V1::id", !"ptr", !"class.sycl::_V1::id", !"ptr", !"ptr", !"struct __wrapper_class"}
!18 = !{!"KernelOne_accTmp3", !"KernelOne_accIn1", !"KernelOne_accIn16", !"KernelOne_accIn2", !"KernelTwo_out", !"KernelTwo_coef"}
!19 = !{!"", !"", !"\00\00\00\00\00\00\00\00", !"", !"", !"\00\00\00\00\01\00\00\00\02\00\00\00\03\00\00\00\04\00\00\00\05\00\00\00\06\00\00\00\07\00\00\00\08\00\00\00\09\00\00\00"}
