; Check IR produced by fusion pass:
; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext\
; RUN: -passes="sycl-kernel-fusion" -S %s\
; RUN: | FileCheck %s --implicit-check-not fused_kernel --check-prefix FUSION

; Check metadata attached to kernel by fusion pass:
; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext\
; RUN: -passes=sycl-kernel-fusion -S %s\
; RUN: | FileCheck %s --check-prefix MD

; Check kernel information produced by fusion pass:
; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext\
; RUN: -passes=sycl-kernel-fusion,print-sycl-module-info -disable-output -S %s\
; RUN: | FileCheck %s --check-prefix INFO

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

%0 = type { %1 }
%1 = type { [1 x i64] }

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0i8(i64 immarg %0, ptr nocapture %1) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0i8(i64 immarg %0, ptr nocapture %1) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 noundef %0) #1

; Function Attrs: nounwind willreturn
declare spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 %0) #2

; Function Attrs: nounwind willreturn
declare spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 %0) #2

; Function Attrs: nounwind willreturn
declare spir_func i64 @_Z29__spirv_BuiltInGlobalLinearIdv() #2

; Function Attrs: nounwind willreturn
declare spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 %0) #2

; Function Attrs: alwaysinline nounwind
define spir_func void @__itt_offload_wi_start_wrapper() #3 {
entry:
  %GroupID = alloca [3 x i64], align 8
  br i1 true, label %return, label %if.end

if.end:                                           ; preds = %entry
  %0 = bitcast ptr %GroupID to ptr
  call void @llvm.lifetime.start.p0i8(i64 24, ptr %0)
  %arrayinit.begin5 = getelementptr inbounds [3 x i64], ptr %GroupID, i64 0, i64 0
  %arrayinit.begin = addrspacecast ptr %arrayinit.begin5 to ptr addrspace(4)
  %1 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 0) #2
  %2 = insertelement <3 x i64> undef, i64 %1, i32 0
  %3 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 1) #2
  %4 = insertelement <3 x i64> %2, i64 %3, i32 1
  %5 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 2) #2
  %6 = insertelement <3 x i64> %4, i64 %5, i32 2
  %7 = extractelement <3 x i64> %6, i32 0
  store i64 %7, ptr addrspace(4) %arrayinit.begin, align 8
  %arrayinit.element6 = getelementptr inbounds [3 x i64], ptr %GroupID, i64 0, i64 1
  %arrayinit.element = addrspacecast ptr %arrayinit.element6 to ptr addrspace(4)
  %8 = extractelement <3 x i64> %6, i32 1
  store i64 %8, ptr addrspace(4) %arrayinit.element, align 8
  %arrayinit.element17 = getelementptr inbounds [3 x i64], ptr %GroupID, i64 0, i64 2
  %arrayinit.element1 = addrspacecast ptr %arrayinit.element17 to ptr addrspace(4)
  %9 = extractelement <3 x i64> %6, i32 2
  store i64 %9, ptr addrspace(4) %arrayinit.element1, align 8
  %10 = call spir_func i64 @_Z29__spirv_BuiltInGlobalLinearIdv() #2
  %11 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 0) #2
  %12 = insertelement <3 x i64> undef, i64 %11, i32 0
  %13 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 1) #2
  %14 = insertelement <3 x i64> %12, i64 %13, i32 1
  %15 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 2) #2
  %16 = insertelement <3 x i64> %14, i64 %15, i32 2
  %17 = extractelement <3 x i64> %16, i32 0
  %18 = extractelement <3 x i64> %16, i32 1
  %mul = mul i64 %17, %18
  %19 = extractelement <3 x i64> %16, i32 2
  %mul2 = mul i64 %mul, %19
  %conv = trunc i64 %mul2 to i32
  call spir_func void @__itt_offload_wi_start_stub(ptr addrspace(4) %arrayinit.begin, i64 %10, i32 %conv) #4
  call void @llvm.lifetime.end.p0i8(i64 24, ptr %0)
  br label %return

return:                                           ; preds = %if.end, %entry
  ret void
}

; Function Attrs: alwaysinline nounwind
define spir_func void @__itt_offload_wi_finish_wrapper() #3 {
entry:
  %GroupID = alloca [3 x i64], align 8
  br i1 true, label %return, label %if.end

if.end:                                           ; preds = %entry
  %0 = bitcast ptr %GroupID to ptr
  call void @llvm.lifetime.start.p0i8(i64 24, ptr %0)
  %arrayinit.begin3 = getelementptr inbounds [3 x i64], ptr %GroupID, i64 0, i64 0
  %arrayinit.begin = addrspacecast ptr %arrayinit.begin3 to ptr addrspace(4)
  %1 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 0) #2
  %2 = insertelement <3 x i64> undef, i64 %1, i32 0
  %3 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 1) #2
  %4 = insertelement <3 x i64> %2, i64 %3, i32 1
  %5 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 2) #2
  %6 = insertelement <3 x i64> %4, i64 %5, i32 2
  %7 = extractelement <3 x i64> %6, i32 0
  store i64 %7, ptr addrspace(4) %arrayinit.begin, align 8
  %arrayinit.element4 = getelementptr inbounds [3 x i64], ptr %GroupID, i64 0, i64 1
  %arrayinit.element = addrspacecast ptr %arrayinit.element4 to ptr addrspace(4)
  %8 = extractelement <3 x i64> %6, i32 1
  store i64 %8, ptr addrspace(4) %arrayinit.element, align 8
  %arrayinit.element15 = getelementptr inbounds [3 x i64], ptr %GroupID, i64 0, i64 2
  %arrayinit.element1 = addrspacecast ptr %arrayinit.element15 to ptr addrspace(4)
  %9 = extractelement <3 x i64> %6, i32 2
  store i64 %9, ptr addrspace(4) %arrayinit.element1, align 8
  %10 = call spir_func i64 @_Z29__spirv_BuiltInGlobalLinearIdv() #2
  call spir_func void @__itt_offload_wi_finish_stub(ptr addrspace(4) %arrayinit.begin, i64 %10) #4
  call void @llvm.lifetime.end.p0i8(i64 24, ptr %0)
  br label %return

return:                                           ; preds = %if.end, %entry
  ret void
}

; Function Attrs: noinline nounwind
declare spir_func void @__itt_offload_wi_finish_stub(ptr addrspace(4) %group_id, i64 %wi_id) #4

; Function Attrs: noinline nounwind
declare spir_func void @__itt_offload_wi_start_stub(ptr addrspace(4) %group_id, i64 %wi_id, i32 %wg_size) #4

; Function Attrs: nounwind
define spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne(ptr addrspace(1) align 4 %_arg_accTmp, ptr byval(%0) align 8 %_arg_accTmp3, ptr addrspace(1) align 4 %_arg_accIn1, ptr byval(%0) align 8 %_arg_accIn16, ptr addrspace(1) align 4 %_arg_accIn2, ptr byval(%0) align 8 %_arg_accIn29) #5 !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_type_qual !9 !kernel_arg_base_type !8 !kernel_arg_name !10 !spirv.ParameterDecorations !11 {
entry:
  call spir_func void @__itt_offload_wi_start_wrapper() #3
  %0 = getelementptr inbounds %0, ptr %_arg_accTmp3, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = load i64, ptr addrspace(4) %1, align 8
  %add.ptr.i = getelementptr inbounds float, ptr addrspace(1) %_arg_accTmp, i64 %2
  %3 = getelementptr inbounds %0, ptr %_arg_accIn16, i64 0, i32 0, i32 0, i64 0
  %4 = addrspacecast ptr %3 to ptr addrspace(4)
  %5 = load i64, ptr addrspace(4) %4, align 8
  %add.ptr.i39 = getelementptr inbounds float, ptr addrspace(1) %_arg_accIn1, i64 %5
  %6 = getelementptr inbounds %0, ptr %_arg_accIn29, i64 0, i32 0, i32 0, i64 0
  %7 = addrspacecast ptr %6 to ptr addrspace(4)
  %8 = load i64, ptr addrspace(4) %7, align 8
  %add.ptr.i53 = getelementptr inbounds float, ptr addrspace(1) %_arg_accIn2, i64 %8
  %9 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0) #2
  %10 = insertelement <3 x i64> undef, i64 %9, i32 0
  %11 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 1) #2
  %12 = insertelement <3 x i64> %10, i64 %11, i32 1
  %13 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 2) #2
  %14 = insertelement <3 x i64> %12, i64 %13, i32 2
  %15 = extractelement <3 x i64> %14, i32 0
  %cmp.i.i = icmp ult i64 %15, 2147483648
  call void @llvm.assume(i1 %cmp.i.i)
  %arrayidx.i.i = getelementptr inbounds float, ptr addrspace(1) %add.ptr.i39, i64 %15
  %arrayidx.ascast.i.i = addrspacecast ptr addrspace(1) %arrayidx.i.i to ptr addrspace(4)
  %16 = load float, ptr addrspace(4) %arrayidx.ascast.i.i, align 4
  %arrayidx.i9.i = getelementptr inbounds float, ptr addrspace(1) %add.ptr.i53, i64 %15
  %arrayidx.ascast.i10.i = addrspacecast ptr addrspace(1) %arrayidx.i9.i to ptr addrspace(4)
  %17 = load float, ptr addrspace(4) %arrayidx.ascast.i10.i, align 4
  %add.i = fadd float %16, %17
  %arrayidx.i13.i = getelementptr inbounds float, ptr addrspace(1) %add.ptr.i, i64 %15
  %arrayidx.ascast.i14.i = addrspacecast ptr addrspace(1) %arrayidx.i13.i to ptr addrspace(4)
  store float %add.i, ptr addrspace(4) %arrayidx.ascast.i14.i, align 4
  call spir_func void @__itt_offload_wi_finish_wrapper() #3
  ret void
}

; Function Attrs: nounwind
define spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo(ptr addrspace(1) align 4 %_arg_accOut, ptr byval(%0) align 8 %_arg_accOut3, ptr addrspace(1) align 4 %_arg_accTmp, ptr byval(%0) align 8 %_arg_accTmp6, ptr addrspace(1) align 4 %_arg_accIn3, ptr byval(%0) align 8 %_arg_accIn39) #5 !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_type_qual !9 !kernel_arg_base_type !8 !kernel_arg_name !17 !spirv.ParameterDecorations !11 {
entry:
  call spir_func void @__itt_offload_wi_start_wrapper() #3
  %0 = getelementptr inbounds %0, ptr %_arg_accOut3, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = load i64, ptr addrspace(4) %1, align 8
  %add.ptr.i = getelementptr inbounds float, ptr addrspace(1) %_arg_accOut, i64 %2
  %3 = getelementptr inbounds %0, ptr %_arg_accTmp6, i64 0, i32 0, i32 0, i64 0
  %4 = addrspacecast ptr %3 to ptr addrspace(4)
  %5 = load i64, ptr addrspace(4) %4, align 8
  %add.ptr.i39 = getelementptr inbounds float, ptr addrspace(1) %_arg_accTmp, i64 %5
  %6 = getelementptr inbounds %0, ptr %_arg_accIn39, i64 0, i32 0, i32 0, i64 0
  %7 = addrspacecast ptr %6 to ptr addrspace(4)
  %8 = load i64, ptr addrspace(4) %7, align 8
  %add.ptr.i53 = getelementptr inbounds float, ptr addrspace(1) %_arg_accIn3, i64 %8
  %9 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0) #2
  %10 = insertelement <3 x i64> undef, i64 %9, i32 0
  %11 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 1) #2
  %12 = insertelement <3 x i64> %10, i64 %11, i32 1
  %13 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 2) #2
  %14 = insertelement <3 x i64> %12, i64 %13, i32 2
  %15 = extractelement <3 x i64> %14, i32 0
  %cmp.i.i = icmp ult i64 %15, 2147483648
  call void @llvm.assume(i1 %cmp.i.i)
  %arrayidx.i.i = getelementptr inbounds float, ptr addrspace(1) %add.ptr.i39, i64 %15
  %arrayidx.ascast.i.i = addrspacecast ptr addrspace(1) %arrayidx.i.i to ptr addrspace(4)
  %16 = load float, ptr addrspace(4) %arrayidx.ascast.i.i, align 4
  %arrayidx.i9.i = getelementptr inbounds float, ptr addrspace(1) %add.ptr.i53, i64 %15
  %arrayidx.ascast.i10.i = addrspacecast ptr addrspace(1) %arrayidx.i9.i to ptr addrspace(4)
  %17 = load float, ptr addrspace(4) %arrayidx.ascast.i10.i, align 4
  %mul.i = fmul float %16, %17
  %arrayidx.i13.i = getelementptr inbounds float, ptr addrspace(1) %add.ptr.i, i64 %15
  %arrayidx.ascast.i14.i = addrspacecast ptr addrspace(1) %arrayidx.i13.i to ptr addrspace(4)
  store float %mul.i, ptr addrspace(4) %arrayidx.ascast.i14.i, align 4
  call spir_func void @__itt_offload_wi_finish_wrapper() #3
  ret void
}

declare !sycl.kernel.fused !18 !sycl.kernel.nd-ranges !20 !sycl.kernel.nd-range !21 void @fused_kernel()

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #2 = { nounwind willreturn }
attributes #3 = { alwaysinline nounwind }
attributes #4 = { noinline nounwind }
attributes #5 = { nounwind }

!6 = !{i32 1, i32 0, i32 1, i32 0, i32 1, i32 0}
!7 = !{!"none", !"none", !"none", !"none", !"none", !"none"}
!8 = !{!"ptr", !"class.sycl::_V1::range", !"ptr", !"class.sycl::_V1::range", !"ptr", !"class.sycl::_V1::range"}
!9 = !{!"", !"", !"", !"", !"", !""}
!10 = !{!"_arg_accTmp", !"_arg_accTmp3", !"_arg_accIn1", !"_arg_accIn16", !"_arg_accIn2", !"_arg_accIn29"}
!11 = !{!12, !14, !12, !14, !12, !14}
!12 = !{!13}
!13 = !{i32 44, i32 4}
!14 = !{!15, !16}
!15 = !{i32 38, i32 2}
!16 = !{i32 44, i32 8}
!17 = !{!"_arg_accOut", !"_arg_accOut3", !"_arg_accTmp", !"_arg_accTmp6", !"_arg_accIn3", !"_arg_accIn39"}
!18 = !{!"fused_0", !19}
!19 = !{!"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne", !"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo"}
!20 = !{!21, !21}
!21 = !{i32 1, !22, !22, !23}
!22 = !{i64 1, i64 1, i64 1}
!23 = !{i64 0, i64 0, i64 0}
!24 = !{
  !"Accessor", !"StdLayout", !"StdLayout", !"StdLayout", !"Accessor",
  !"StdLayout", !"StdLayout", !"StdLayout", !"Accessor", !"StdLayout",
  !"StdLayout", !"StdLayout"}
!25 = !{i8 1, i8 0, i8 0, i8 1, i8 1, i8 0, i8 0, i8 1, i8 1, i8 0, i8 0, i8 1}
!26 = !{!"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne", !24, !25}
!27 = !{!"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo", !24, !25}
!sycl.moduleinfo = !{!26, !27}

; Test scenario: Fusion of two kernels with mo identical parameters.


; This prefix focuses on the correct signature of the fused kernel and presence
; of core operations of the two input kernels in the body of the fused kernel.
; Also verifies that the stub function for fusion has been removed 
; via 'implicit-check-not'.

; FUSION-LABEL: define spir_kernel void @fused_0
; FUSION-SAME: ptr addrspace(1) align 4 
; FUSION-SAME: ptr byval(%0) align 8 
; FUSION-SAME: ptr addrspace(1) align 4 
; FUSION-SAME: ptr byval(%0) align 8 
; FUSION-SAME: ptr addrspace(1) align 4 
; FUSION-SAME: ptr byval(%0) align 8 
; FUSION-SAME: ptr addrspace(1) align 4 
; FUSION-SAME: ptr byval(%0) align 8 
; FUSION-SAME: ptr addrspace(1) align 4 
; FUSION-SAME: ptr byval(%0) align 8 
; FUSION-SAME: ptr addrspace(1) align 4 
; FUSION-SAME: ptr byval(%0) align 8
; FUSION-LABEL: entry:
; FUSION-NEXT: call spir_func void @__itt_offload_wi_start_wrapper()
; FUSION:   [[IN1:%.*]] = load float
; FUSION:   [[IN2:%.*]] = load float
; FUSION:   [[ADD:%.*]] = fadd float [[IN1]], [[IN2]]
; FUSION:   store float [[ADD]]
; FUSION-LABEL:   call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 784)
; FUSION:   [[IN3:%.*]] = load float
; FUSION:   [[IN4:%.*]] = load float
; FUSION:   [[MUL:%.*]] = fmul float [[IN3]], [[IN4]]
; FUSION:   store float [[MUL]]
; FUSION:   call spir_func void @__itt_offload_wi_finish_wrapper()
; FUSION-NEXT:   ret void

; This prefix focuses on the correct 'kernel_arg_*' metadata being 
; attached to the fused kernel.

; MD-LABEL: define spir_kernel void @fused_0
; MD-SAME: ptr addrspace(1) align 4  %[[ARG1:[^,]+]]
; MD-SAME: ptr byval(%0) align 8 %[[ARG2:[^,]+]]
; MD-SAME: ptr addrspace(1) align 4 %[[ARG3:[^,]+]]
; MD-SAME: ptr byval(%0) align 8 %[[ARG4:[^,]+]]
; MD-SAME: ptr addrspace(1) align 4 %[[ARG5:[^,]+]]
; MD-SAME: ptr byval(%0) align 8 %[[ARG6:[^,]+]]
; MD-SAME: ptr addrspace(1) align 4 %[[ARG7:[^,]+]]
; MD-SAME: ptr byval(%0) align 8 %[[ARG8:[^,]+]]
; MD-SAME: ptr addrspace(1) align 4 %[[ARG9:[^,]+]]
; MD-SAME: ptr byval(%0) align 8 %[[ARG10:[^,]+]]
; MD-SAME: ptr addrspace(1) align 4 %[[ARG11:[^,]+]]
; MD-SAME: ptr byval(%0) align 8 %[[ARG12:[^)]+]]
; MD-SAME: !kernel_arg_addr_space ![[#ADDR_SPACE:]] 
; MD-SAME: !kernel_arg_access_qual ![[#ACCESS_QUAL:]]
; MD-SAME: !kernel_arg_type ![[#ARG_TYPE:]]
; MD-SAME: !kernel_arg_type_qual ![[#TYPE_QUAL:]] 
; MD-SAME: !kernel_arg_base_type ![[#ARG_TYPE]] 
; MD-SAME: !kernel_arg_name ![[#ARG_NAME:]] 
;.
; MD: [[#ADDR_SPACE]] = !{i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0}
; MD: [[#ACCESS_QUAL]] = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
; MD: [[#ARG_TYPE]] = !{!"ptr", !"class.sycl::_V1::range", !"ptr", !"class.sycl::_V1::range", !"ptr", !"class.sycl::_V1::range", !"ptr", !"class.sycl::_V1::range", !"ptr", !"class.sycl::_V1::range", !"ptr", !"class.sycl::_V1::range"}
; MD: [[#TYPE_QUAL]] = !{!"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !""}
; MD: [[#ARG_NAME]] = !{!"[[ARG1]]", !"[[ARG2]]", !"[[ARG3]]", !"[[ARG4]]", !"[[ARG5]]", !"[[ARG6]]", !"[[ARG7]]", !"[[ARG8]]", !"[[ARG9]]", !"[[ARG10]]", !"[[ARG11]]", !"[[ARG12]]"}
;.

; This prefix focuses on the correct update of the SYCLModuleInfo, 
; tested by verifying the textual dump of the module/kernel info.

; INFO-LABEL: KernelName: fused_0
; INFO-NEXT:    Args:
; INFO-NEXT:      Kinds: Accessor, StdLayout, StdLayout, StdLayout, Accessor,
; INFO-SAME:             StdLayout, StdLayout, StdLayout, Accessor, StdLayout,
; INFO-SAME:             StdLayout, StdLayout, Accessor, StdLayout, StdLayout,
; INFO-SAME:             StdLayout, Accessor, StdLayout, StdLayout, StdLayout,
; INFO-SAME:             Accessor, StdLayout, StdLayout, StdLayout
; INFO-NEXT:      Mask: 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
; INFO-SAME:            1, 1, 0, 0, 1
