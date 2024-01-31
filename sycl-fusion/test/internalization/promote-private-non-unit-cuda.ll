; REQUIRES: cuda
; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext \
; RUN: -passes=sycl-internalization -S %s | FileCheck %s

; This test is a reduced IR version of
; sycl/test-e2e/KernelFusion/internalize_non_unit_localsize.cpp for CUDA

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }
%struct.MyStruct = type { i32, %"class.sycl::_V1::vec" }
%"class.sycl::_V1::vec" = type { <3 x i32> }

declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
declare ptr @llvm.nvvm.implicit.offset() #1

define void @fused_0(ptr addrspace(1) nocapture noundef align 16 %KernelOne__arg_accTmp,
    ptr nocapture noundef readonly byval(%"class.sycl::_V1::range") align 8 %KernelOne__arg_accTmp3,
    ptr addrspace(1) nocapture noundef readonly align 4 %KernelOne__arg_accIn,
    ptr nocapture noundef readonly byval(%"class.sycl::_V1::range") align 8 %KernelOne__arg_accIn6,
    ptr addrspace(1) nocapture noundef align 1 %KernelOne__arg_accTmp27,
    ptr nocapture noundef readonly byval(%"class.sycl::_V1::range") align 8 %KernelOne__arg_accTmp210,
    ptr addrspace(1) nocapture noundef writeonly align 4 %KernelTwo__arg_accOut,
    ptr nocapture noundef readonly byval(%"class.sycl::_V1::range") align 8 %KernelTwo__arg_accOut3)
      local_unnamed_addr #3 !sycl.kernel.promote !13 !sycl.kernel.promote.localsize !14 !sycl.kernel.promote.elemsize !15 {
; CHECK-LABEL: define void @fused_0(
; CHECK-SAME: ptr nocapture noundef readonly byval(%"class.sycl::_V1::range") align 8 [[KERNELONE__ARG_ACCTMP3:%[^,]*accTmp3]],
; CHECK-SAME: ptr nocapture noundef readonly byval(%"class.sycl::_V1::range") align 8 [[KERNELONE__ARG_ACCTMP210:%[^,]*accTmp210]]
; CHECK:         entry:
; CHECK:           [[TMP0:%.*]] = alloca i8, i64 3, align 1
; CHECK:           [[TMP1:%.*]] = alloca i8, i64 96, align 16
; CHECK:           [[KERNELONE__ARG_ACCTMP2103_SROA_0_0_COPYLOAD:%.*]] = load i64, ptr [[KERNELONE__ARG_ACCTMP210]], align 8
; CHECK:           [[KERNELONE__ARG_ACCTMP31_SROA_0_0_COPYLOAD:%.*]] = load i64, ptr [[KERNELONE__ARG_ACCTMP3]], align 8
; CHECK:           [[TMP2:%.*]] = urem i64 [[KERNELONE__ARG_ACCTMP31_SROA_0_0_COPYLOAD]], 3
; CHECK:           [[TMP3:%.*]] = urem i64 [[KERNELONE__ARG_ACCTMP2103_SROA_0_0_COPYLOAD]], 3
; CHECK:           [[MUL:%.*]] = mul nuw nsw i64 [[GLOBAL_ID:.*]], 3
; CHECK:           [[ADD:%.*]] = add nuw nsw i64 [[MUL]], 1
; CHECK:           [[TMP10:%.*]] = add i64 [[TMP2]], [[ADD]]
; CHECK:           [[TMP11:%.*]] = urem i64 [[TMP10]], 3
; CHECK:           [[ARRAYIDX_1:%.*]] = getelementptr inbounds %struct.MyStruct, ptr [[TMP1]], i64 [[TMP11]]

; COM:             This i8-GEP _was_ not remapped because it addresses into a single MyStruct element
; CHECK:           [[ARRAYIDX_2:%.*]] = getelementptr inbounds i8, ptr [[ARRAYIDX_1]], i64 20
; CHECK:           store i32 {{.*}}, ptr [[ARRAYIDX_2]], align 4
; CHECK:           [[TMP12:%.*]] = add i64 [[TMP3]], [[ADD]]
; CHECK:           [[TMP13:%.*]] = urem i64 [[TMP12]], 3

; COM:             This i8-GEP was remapped because it selects an element of the underlying i8-buffer
; CHECK:           [[ARRAYIDX_3:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP13]]

; CHECK:           store i8 {{.*}}, ptr [[ARRAYIDX_3]], align 1
; CHECK:           store i32 {{.*}}, ptr addrspace(1)
; CHECK:           ret void
;
entry:
  %KernelOne__arg_accTmp2103.sroa.0.0.copyload = load i64, ptr %KernelOne__arg_accTmp210, align 8
  %KernelOne__arg_accIn62.sroa.0.0.copyload = load i64, ptr %KernelOne__arg_accIn6, align 8
  %KernelOne__arg_accTmp31.sroa.0.0.copyload = load i64, ptr %KernelOne__arg_accTmp3, align 8
  %add.ptr.j2 = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %KernelOne__arg_accTmp, i64 %KernelOne__arg_accTmp31.sroa.0.0.copyload
  %add.ptr.i37.i = getelementptr inbounds i32, ptr addrspace(1) %KernelOne__arg_accIn, i64 %KernelOne__arg_accIn62.sroa.0.0.copyload
  %add.ptr.i43.i = getelementptr inbounds i8, ptr addrspace(1) %KernelOne__arg_accTmp27, i64 %KernelOne__arg_accTmp2103.sroa.0.0.copyload
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %conv.i1.j7 = sext i32 %0 to i64
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %conv.i3.j7 = sext i32 %1 to i64
  %mul.j7 = mul nsw i64 %conv.i3.j7, %conv.i1.j7
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %conv.i2.j7 = sext i32 %2 to i64
  %add.j7 = add nsw i64 %mul.j7, %conv.i2.j7
  %3 = tail call ptr @llvm.nvvm.implicit.offset()
  %4 = load i32, ptr %3, align 4
  %conv.j8 = zext i32 %4 to i64
  %add4.j7 = add nsw i64 %add.j7, %conv.j8
  %mul.j2 = mul nuw nsw i64 %add4.j7, 3
  %add.j2 = add nuw nsw i64 %mul.j2, 1
  %arrayidx.j2 = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i37.i, i64 %add.j2
  %5 = load i32, ptr addrspace(1) %arrayidx.j2, align 4
  %arrayidx.i55.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.j2, i64 %add.j2
  %arrayidx.j3 = getelementptr inbounds i8, ptr addrspace(1) %arrayidx.i55.i, i64 20
  store i32 %5, ptr addrspace(1) %arrayidx.j3, align 4
  %conv.j2 = trunc i32 %5 to i8
  %arrayidx.i73.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i43.i, i64 %add.j2
  store i8 %conv.j2, ptr addrspace(1) %arrayidx.i73.i, align 1
  %KernelTwo__arg_accOut34.sroa.0.0.copyload = load i64, ptr %KernelTwo__arg_accOut3, align 8
  %add.ptr.i.i7 = getelementptr inbounds i32, ptr addrspace(1) %KernelTwo__arg_accOut, i64 %KernelTwo__arg_accOut34.sroa.0.0.copyload
  %6 = load i32, ptr %3, align 4
  %conv.j7.i13 = zext i32 %6 to i64
  %add4.j6.i14 = add nsw i64 %add.j7, %conv.j7.i13
  %mul.i.i16 = mul nuw nsw i64 %add4.j6.i14, 3
  %add.i45.i = add nuw nsw i64 %mul.i.i16, 1
  %arrayidx.i.i17 = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.j2, i64 %add.i45.i
  %arrayidx.j2.i19 = getelementptr inbounds i8, ptr addrspace(1) %arrayidx.i.i17, i64 20
  %7 = load i32, ptr addrspace(1) %arrayidx.j2.i19, align 4
  %arrayidx.i55.i20 = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i43.i, i64 %add.i45.i
  %8 = load i8, ptr addrspace(1) %arrayidx.i55.i20, align 1
  %conv.i.i22 = sext i8 %8 to i32
  %add.i.i23 = add nsw i32 %7, %conv.i.i22
  %arrayidx.i64.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i.i7, i64 %add.i45.i
  store i32 %add.i.i23, ptr addrspace(1) %arrayidx.i64.i, align 4
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nofree nosync nounwind speculatable memory(none) }
attributes #3 = { nofree nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: write) "frame-pointer"="all" "target-cpu"="sm_80" "target-features"="+ptx82,+sm_80" "uniform-work-group-size"="true" }

!nvvm.annotations = !{!10}

!10 = !{ptr @fused_0, !"kernel", i32 1}
!13 = !{!"private", !"none", !"none", !"none", !"private", !"none", !"none", !"none"}
!14 = !{i64 3, !"", !"", !"", i64 3, !"", !"", !""}
!15 = !{i64 32, !"", !"", !"", i64 1, !"", !"", !""}
