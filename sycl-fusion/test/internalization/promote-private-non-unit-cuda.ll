; REQUIRES: cuda
; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext \
; RUN: -passes=sycl-internalization --sycl-info-path %S/../kernel-fusion/kernel-info.yaml -S %s | FileCheck %s

; This test is a reduced IR version of
; sycl/test-e2e/KernelFusion/internalize_non_unit_localsize.cpp for CUDA

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }
%struct.MyStruct = type { i32, %"class.sycl::_V1::vec" }
%"class.sycl::_V1::vec" = type { <3 x i32> }

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: nofree nosync nounwind speculatable memory(none)
declare ptr @llvm.nvvm.implicit.offset() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #2

; Function Attrs: nofree nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: write)
define void @fused_0(ptr addrspace(1) nocapture noundef align 16 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp,
    ptr nocapture noundef readonly byval(%"class.sycl::_V1::range") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp3,
    ptr addrspace(1) nocapture noundef readonly align 4 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn,
    ptr nocapture noundef readonly byval(%"class.sycl::_V1::range") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn6,
    ptr addrspace(1) nocapture noundef align 1 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp27,
    ptr nocapture noundef readonly byval(%"class.sycl::_V1::range") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp210,
    ptr addrspace(1) nocapture noundef writeonly align 4 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut,
    ptr nocapture noundef readonly byval(%"class.sycl::_V1::range") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut3)
      local_unnamed_addr #3 !kernel_arg_buffer_location !11 !kernel_arg_runtime_aligned !12 !kernel_arg_exclusive_ptr !12 !sycl.kernel.promote !13 !sycl.kernel.promote.localsize !14 !sycl.kernel.promote.elemsize !15 !sycl.kernel.constants !16 {
; CHECK-LABEL: define void @fused_0(
; CHECK-SAME: ptr nocapture noundef readonly byval(%"class.sycl::_V1::range") align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP3:%[^,]*accTmp3]],
; CHECK-SAME: ptr nocapture noundef readonly byval(%"class.sycl::_V1::range") align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP210:%[^,]*accTmp210]]
; CHECK:         entry:
; CHECK:           [[TMP0:%.*]] = alloca i8, i64 3, align 1
; CHECK:           [[TMP1:%.*]] = alloca i8, i64 96, align 16
; CHECK:           [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP2103_SROA_0_0_COPYLOAD:%.*]] = load i64, ptr [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP210]], align 8
; CHECK:           [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP31_SROA_0_0_COPYLOAD:%.*]] = load i64, ptr [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP3]], align 8
; CHECK:           [[TMP2:%.*]] = urem i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP31_SROA_0_0_COPYLOAD]], 3
; CHECK:           [[TMP3:%.*]] = urem i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP2103_SROA_0_0_COPYLOAD]], 3
; CHECK:           [[CMP_I_I_I_I:%.*]] = icmp ult i64 [[GLOBAL_ID:%.*]], 2147483648
; CHECK:           tail call void @llvm.assume(i1 [[CMP_I_I_I_I]])
; CHECK:           [[MUL_I_I:%.*]] = mul nuw nsw i64 [[GLOBAL_ID]], 3
; CHECK:           [[ADD_I_I:%.*]] = add nuw nsw i64 [[MUL_I_I]], 1
; CHECK:           [[TMP10:%.*]] = add i64 [[TMP2]], [[ADD_I_I]]
; CHECK:           [[TMP11:%.*]] = urem i64 [[TMP10]], 3
; CHECK:           [[ARRAYIDX_I55_I:%.*]] = getelementptr inbounds %struct.MyStruct, ptr [[TMP1]], i64 [[TMP11]]

; COM:             This i8-GEP _was_ not remapped because it addresses into a single MyStruct element
; CHECK:           [[ARRAYIDX_I_I_I:%.*]] = getelementptr inbounds i8, ptr [[ARRAYIDX_I55_I]], i64 20
; CHECK:           store i32 {{.*}}, ptr [[ARRAYIDX_I_I_I]], align 4
; CHECK:           [[TMP12:%.*]] = add i64 [[TMP3]], [[ADD_I_I]]
; CHECK:           [[TMP13:%.*]] = urem i64 [[TMP12]], 3

; COM:             This i8-GEP was remapped because it selects an element of the underlying i8-buffer
; CHECK:           [[ARRAYIDX_I73_I:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP13]]

; CHECK:           store i8 {{.*}}, ptr [[ARRAYIDX_I73_I]], align 1
; CHECK:           store i32 {{.*}}, ptr addrspace(1)
; CHECK:           ret void
;
entry:
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp2103.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp210, align 8
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn62.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn6, align 8
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp31.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp3, align 8
  %add.ptr.i.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp31.sroa.0.0.copyload
  %add.ptr.i37.i = getelementptr inbounds i32, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn62.sroa.0.0.copyload
  %add.ptr.i43.i = getelementptr inbounds i8, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp27, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp2103.sroa.0.0.copyload
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %conv.i1.i.i.i.i.i.i.i = sext i32 %0 to i64
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %conv.i3.i.i.i.i.i.i.i = sext i32 %1 to i64
  %mul.i.i.i.i.i.i.i = mul nsw i64 %conv.i3.i.i.i.i.i.i.i, %conv.i1.i.i.i.i.i.i.i
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %conv.i2.i.i.i.i.i.i.i = sext i32 %2 to i64
  %add.i.i.i.i.i.i.i = add nsw i64 %mul.i.i.i.i.i.i.i, %conv.i2.i.i.i.i.i.i.i
  %3 = tail call ptr @llvm.nvvm.implicit.offset()
  %4 = load i32, ptr %3, align 4, !tbaa !17
  %conv.i.i.i.i.i.i.i.i = zext i32 %4 to i64
  %add4.i.i.i.i.i.i.i = add nsw i64 %add.i.i.i.i.i.i.i, %conv.i.i.i.i.i.i.i.i
  %cmp.i.i.i.i = icmp ult i64 %add4.i.i.i.i.i.i.i, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i.i.i)
  %mul.i.i = mul nuw nsw i64 %add4.i.i.i.i.i.i.i, 3
  %add.i.i = add nuw nsw i64 %mul.i.i, 1
  %arrayidx.i.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i37.i, i64 %add.i.i
  %5 = load i32, ptr addrspace(1) %arrayidx.i.i, align 4, !tbaa !21
  %arrayidx.i55.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i.i, i64 %add.i.i
  %arrayidx.i.i.i = getelementptr inbounds i8, ptr addrspace(1) %arrayidx.i55.i, i64 20
  store i32 %5, ptr addrspace(1) %arrayidx.i.i.i, align 4, !tbaa !21
  %conv.i.i = trunc i32 %5 to i8
  %arrayidx.i73.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i43.i, i64 %add.i.i
  store i8 %conv.i.i, ptr addrspace(1) %arrayidx.i73.i, align 1, !tbaa !25
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut34.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut3, align 8
  %add.ptr.i.i7 = getelementptr inbounds i32, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut34.sroa.0.0.copyload
  %6 = load i32, ptr %3, align 4, !tbaa !17
  %conv.i.i.i.i.i.i.i.i13 = zext i32 %6 to i64
  %add4.i.i.i.i.i.i.i14 = add nsw i64 %add.i.i.i.i.i.i.i, %conv.i.i.i.i.i.i.i.i13
  %cmp.i.i.i.i15 = icmp ult i64 %add4.i.i.i.i.i.i.i14, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i.i.i15)
  %mul.i.i16 = mul nuw nsw i64 %add4.i.i.i.i.i.i.i14, 3
  %add.i45.i = add nuw nsw i64 %mul.i.i16, 1
  %arrayidx.i.i17 = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i.i, i64 %add.i45.i
  %arrayidx.i.i.i19 = getelementptr inbounds i8, ptr addrspace(1) %arrayidx.i.i17, i64 20
  %7 = load i32, ptr addrspace(1) %arrayidx.i.i.i19, align 4, !tbaa !21
  %arrayidx.i55.i20 = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i43.i, i64 %add.i45.i
  %8 = load i8, ptr addrspace(1) %arrayidx.i55.i20, align 1, !tbaa !25
  %conv.i.i22 = sext i8 %8 to i32
  %add.i.i23 = add nsw i32 %7, %conv.i.i22
  %arrayidx.i64.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i.i7, i64 %add.i45.i
  store i32 %add.i.i23, ptr addrspace(1) %arrayidx.i64.i, align 4, !tbaa !21
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nofree nosync nounwind speculatable memory(none) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #3 = { nofree nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: write) "frame-pointer"="all" "target-cpu"="sm_80" "target-features"="+ptx82,+sm_80" "uniform-work-group-size"="true" }

!opencl.spir.version = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!spirv.Source = !{!1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1}
!nvvmir.version = !{!4, !4}
!llvm.module.flags = !{!5, !6, !7, !8, !9}
!sycl.specialization-constants = !{}
!sycl.specialization-constants-default-values = !{}
!nvvm.annotations = !{!10}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 18.0.0git (git@github.com:jopperm/llvm.git ead5ebd9efe317a292af33fabb67095e6e02e502)"}
!3 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!4 = !{i32 2, i32 0}
!5 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 2]}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{i32 7, !"nvvm-reflect-ftz", i32 0}
!8 = !{i32 7, !"nvvm-reflect-prec-sqrt", i32 0}
!9 = !{i32 7, !"frame-pointer", i32 2}
!10 = !{ptr @fused_0, !"kernel", i32 1}
!11 = !{i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1}
!12 = !{i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false}
!13 = !{!"private", !"none", !"none", !"none", !"private", !"none", !"none", !"none"}
!14 = !{i64 3, !"", !"", !"", i64 3, !"", !"", !""}
!15 = !{i64 32, !"", !"", !"", i64 1, !"", !"", !""}
!16 = !{!"", !"", !"", !"\00\00\00\00\00\00\00\00", !"", !"", !"", !"\00\00\00\00\00\00\00\00"}
!17 = !{!18, !18, i64 0}
!18 = !{!"int", !19, i64 0}
!19 = !{!"omnipotent char", !20, i64 0}
!20 = !{!"Simple C/C++ TBAA"}
!21 = !{!22, !22, i64 0}
!22 = !{!"int", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!23, !23, i64 0}
