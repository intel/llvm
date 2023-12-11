; REQUIRES: hip_amd
; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext \
; RUN: -passes=sycl-internalization --sycl-info-path %S/../kernel-fusion/kernel-info.yaml -S %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8"
target triple = "amdgcn-amd-amdhsa"

%struct.MyStruct = type { i32, %"class.sycl::_V1::vec" }
%"class.sycl::_V1::vec" = type { <3 x i32> }

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.amdgcn.workgroup.id.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare align 4 ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.amdgcn.workitem.id.x() #0

; Function Attrs: nounwind speculatable memory(none)
declare ptr addrspace(5) @llvm.amdgcn.implicit.offset() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef %0) #2

define amdgpu_kernel void @fused_0(ptr addrspace(1) noundef align 16 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp31, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp.coerce, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp.coerce3, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp.coerce7, ptr addrspace(1) noundef align 4 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn35, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn.coerce, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn.coerce13, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn.coerce17, ptr addrspace(1) noundef align 1 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp239, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp2.coerce, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp2.coerce24, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp2.coerce28, ptr addrspace(1) noundef align 4 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut40, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut.coerce, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut.coerce3, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut.coerce7, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn.coerce, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn.coerce23, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn.coerce27) #3 !kernel_arg_buffer_location !10 !kernel_arg_runtime_aligned !11 !kernel_arg_exclusive_ptr !11 !sycl.kernel.promote !12 !sycl.kernel.promote.localsize !13 !sycl.kernel.promote.elemsize !14 !sycl.kernel.constants !15 {
; CHECK-LABEL: define amdgpu_kernel void @fused_0(
; CHECK-SAME:      i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP_COERCE:%.*]], i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP_COERCE3:%.*]], i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP_COERCE7:%.*]], ptr addrspace(1) noundef align 4 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCIN35:%.*]], i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCIN_COERCE:%.*]], i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCIN_COERCE13:%.*]], i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCIN_COERCE17:%.*]], i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP2_COERCE:%.*]], i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP2_COERCE24:%.*]], i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP2_COERCE28:%.*]], ptr addrspace(1) noundef align 4 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE0_CLES2_E9KERNELTWO__ARG_ACCOUT40:%.*]], i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE0_CLES2_E9KERNELTWO__ARG_ACCOUT_COERCE:%.*]], i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE0_CLES2_E9KERNELTWO__ARG_ACCOUT_COERCE3:%.*]], i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE0_CLES2_E9KERNELTWO__ARG_ACCOUT_COERCE7:%.*]], i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE0_CLES2_E9KERNELTWO__ARG_ACCIN_COERCE:%.*]], i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE0_CLES2_E9KERNELTWO__ARG_ACCIN_COERCE23:%.*]], i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE0_CLES2_E9KERNELTWO__ARG_ACCIN_COERCE27:%.*]]) #[[ATTR3:[0-9]+]] !kernel_arg_buffer_location !10 !kernel_arg_runtime_aligned !11 !kernel_arg_exclusive_ptr !11 !sycl.kernel.constants [[META12:![0-9]+]] {
; CHECK:         entry:
; CHECK:           [[TMP0:%.*]] = alloca i8, i64 3, align 1, addrspace(5)
; CHECK:           [[TMP1:%.*]] = alloca i8, i64 96, align 16, addrspace(5)
; CHECK:           [[TMP2:%.*]] = urem i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP_COERCE7]], 3
; CHECK:           [[ADD_PTR_I_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT:%.*]], ptr addrspace(5) [[TMP1]], i64 [[TMP2]]
; CHECK:           [[TMP3:%.*]] = urem i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP2_COERCE28]], 3
; CHECK:           [[ADD_PTR_I85_I:%.*]] = getelementptr inbounds i8, ptr addrspace(5) [[TMP0]], i64 [[TMP3]]
; CHECK:           [[TMP12:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[V_I_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr addrspace(5) [[TMP1]], i64 [[TMP12]], i32 1
; CHECK:           [[ARRAYIDX_I_I_I:%.*]] = getelementptr inbounds i32, ptr addrspace(5) [[V_I_I]], i64 1
; CHECK:           [[TMP15:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[V46_I_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr addrspace(5) [[TMP1]], i64 [[TMP15]], i32 1
; CHECK:           [[ARRAYIDX_I_I102_I:%.*]] = getelementptr inbounds i32, ptr addrspace(5) [[V46_I_I]], i64 1
; CHECK:           [[TMP18:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[V76_I_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr addrspace(5) [[TMP1]], i64 [[TMP18]], i32 1
; CHECK:           [[ARRAYIDX_I_I115_I:%.*]] = getelementptr inbounds i32, ptr addrspace(5) [[V76_I_I]], i64 1
; CHECK:           [[TMP22:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I124_I:%.*]] = getelementptr inbounds i8, ptr addrspace(5) [[TMP0]], i64 [[TMP22]]
; CHECK:           [[TMP26:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I136_I:%.*]] = getelementptr inbounds i8, ptr addrspace(5) [[TMP0]], i64 [[TMP26]]
; CHECK:           [[TMP30:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I146_I:%.*]] = getelementptr inbounds i8, ptr addrspace(5) [[TMP0]], i64 [[TMP30]]
; CHECK:           [[TMP31:%.*]] = urem i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP_COERCE7]], 3
; CHECK:           [[ADD_PTR_I107_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr addrspace(5) [[TMP1]], i64 [[TMP31]]
; CHECK:           [[TMP32:%.*]] = urem i64 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP2_COERCE28]], 3
; CHECK:           [[ADD_PTR_I113_I:%.*]] = getelementptr inbounds i8, ptr addrspace(5) [[TMP0]], i64 [[TMP32]]
; CHECK:           [[TMP40:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[V_I_I11:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr addrspace(5) [[TMP1]], i64 [[TMP40]], i32 1
; CHECK:           [[ARRAYIDX_I_I_I12:%.*]] = getelementptr inbounds i32, ptr addrspace(5) [[V_I_I11]], i64 1
; CHECK:           [[TMP44:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I124_I14:%.*]] = getelementptr inbounds i8, ptr addrspace(5) [[TMP0]], i64 [[TMP44]]
; CHECK:           [[TMP47:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[V53_I_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr addrspace(5) [[TMP1]], i64 [[TMP47]], i32 1
; CHECK:           [[ARRAYIDX_I_I136_I:%.*]] = getelementptr inbounds i32, ptr addrspace(5) [[V53_I_I]], i64 1
; CHECK:           [[TMP51:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I147_I:%.*]] = getelementptr inbounds i8, ptr addrspace(5) [[TMP0]], i64 [[TMP51]]
; CHECK:           [[TMP54:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[V114_I_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr addrspace(5) [[TMP1]], i64 [[TMP54]], i32 1
; CHECK:           [[ARRAYIDX_I_I161_I:%.*]] = getelementptr inbounds i32, ptr addrspace(5) [[V114_I_I]], i64 1
; CHECK:           [[TMP58:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I170_I:%.*]] = getelementptr inbounds i8, ptr addrspace(5) [[TMP0]], i64 [[TMP58]]
; CHECK:           ret void
;
entry:
  %add.ptr.i.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp31, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp.coerce7
  %add.ptr.i82.i = getelementptr inbounds i32, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn35, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn.coerce17
  %add.ptr.i85.i = getelementptr inbounds i8, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp239, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp2.coerce28
  %0 = call i32 @llvm.amdgcn.workgroup.id.x()
  %conv.i1.i.i.i.i.i.i.i = zext i32 %0 to i64
  %1 = call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %arrayidx.i.i.i.i.i.i.i.i = getelementptr inbounds i16, ptr addrspace(4) %1, i64 2
  %2 = load i16, ptr addrspace(4) %arrayidx.i.i.i.i.i.i.i.i, align 4, !tbaa !16
  %conv.i.i.i.i.i.i.i.i = zext i16 %2 to i64
  %mul.i.i.i.i.i.i.i = mul nuw nsw i64 %conv.i.i.i.i.i.i.i.i, %conv.i1.i.i.i.i.i.i.i
  %3 = call i32 @llvm.amdgcn.workitem.id.x(), !range !20, !noundef !21
  %conv.i2.i.i.i.i.i.i.i = zext nneg i32 %3 to i64
  %add.i.i.i.i.i.i.i = add nuw nsw i64 %mul.i.i.i.i.i.i.i, %conv.i2.i.i.i.i.i.i.i
  %4 = call ptr addrspace(5) @llvm.amdgcn.implicit.offset()
  %5 = load i32, ptr addrspace(5) %4, align 4
  %zext.i.i.i.i.i.i.i.i = zext i32 %5 to i64
  %add4.i.i.i.i.i.i.i = add nuw nsw i64 %add.i.i.i.i.i.i.i, %zext.i.i.i.i.i.i.i.i
  %cmp.i.i.i = icmp ult i64 %add4.i.i.i.i.i.i.i, 2147483648
  call void @llvm.assume(i1 %cmp.i.i.i)
  %mul.i.i = mul nuw nsw i64 %add4.i.i.i.i.i.i.i, 3
  %arrayidx.i.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i82.i, i64 %mul.i.i
  %6 = load i32, ptr addrspace(1) %arrayidx.i.i, align 4, !tbaa !22
  %v.i.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i.i, i64 %mul.i.i, i32 1
  %arrayidx.i.i.i = getelementptr inbounds i32, ptr addrspace(1) %v.i.i, i64 1
  store i32 %6, ptr addrspace(1) %arrayidx.i.i.i, align 4, !tbaa !22
  %add.i.i = add nuw nsw i64 %mul.i.i, 1
  %arrayidx.i94.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i82.i, i64 %add.i.i
  %7 = load i32, ptr addrspace(1) %arrayidx.i94.i, align 4, !tbaa !22
  %v46.i.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i.i, i64 %add.i.i, i32 1
  %arrayidx.i.i102.i = getelementptr inbounds i32, ptr addrspace(1) %v46.i.i, i64 1
  store i32 %7, ptr addrspace(1) %arrayidx.i.i102.i, align 4, !tbaa !22
  %add.i106.i = add nuw nsw i64 %mul.i.i, 2
  %arrayidx.i107.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i82.i, i64 %add.i106.i
  %8 = load i32, ptr addrspace(1) %arrayidx.i107.i, align 4, !tbaa !22
  %v76.i.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i.i, i64 %add.i106.i, i32 1
  %arrayidx.i.i115.i = getelementptr inbounds i32, ptr addrspace(1) %v76.i.i, i64 1
  store i32 %8, ptr addrspace(1) %arrayidx.i.i115.i, align 4, !tbaa !22
  %9 = load i32, ptr addrspace(1) %arrayidx.i.i, align 4, !tbaa !22
  %10 = trunc i32 %9 to i8
  %conv.i.i = xor i8 %10, -86
  %arrayidx.i124.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i85.i, i64 %add.i106.i
  store i8 %conv.i.i, ptr addrspace(1) %arrayidx.i124.i, align 1, !tbaa !26
  %11 = load i32, ptr addrspace(1) %arrayidx.i94.i, align 4, !tbaa !22
  %12 = trunc i32 %11 to i8
  %conv115.i.i = xor i8 %12, -86
  %arrayidx.i136.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i85.i, i64 %add.i.i
  store i8 %conv115.i.i, ptr addrspace(1) %arrayidx.i136.i, align 1, !tbaa !26
  %13 = load i32, ptr addrspace(1) %arrayidx.i107.i, align 4, !tbaa !22
  %14 = trunc i32 %13 to i8
  %conv145.i.i = xor i8 %14, -86
  %arrayidx.i146.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i85.i, i64 %mul.i.i
  store i8 %conv145.i.i, ptr addrspace(1) %arrayidx.i146.i, align 1, !tbaa !26
  %add.ptr.i.i1 = getelementptr inbounds i32, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut40, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut.coerce7
  %add.ptr.i107.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp31, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp.coerce7
  %add.ptr.i110.i = getelementptr inbounds i32, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn35, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn.coerce27
  %add.ptr.i113.i = getelementptr inbounds i8, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp239, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp2.coerce28
  %15 = call i32 @llvm.amdgcn.workgroup.id.x()
  %conv.i1.i.i.i.i.i.i.i2 = zext i32 %15 to i64
  %16 = call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %arrayidx.i.i.i.i.i.i.i.i3 = getelementptr inbounds i16, ptr addrspace(4) %16, i64 2
  %17 = load i16, ptr addrspace(4) %arrayidx.i.i.i.i.i.i.i.i3, align 4, !tbaa !16
  %conv.i.i.i.i.i.i.i.i4 = zext i16 %17 to i64
  %mul.i.i.i.i.i.i.i5 = mul nuw nsw i64 %conv.i.i.i.i.i.i.i.i4, %conv.i1.i.i.i.i.i.i.i2
  %18 = call i32 @llvm.amdgcn.workitem.id.x(), !range !20, !noundef !21
  %conv.i2.i.i.i.i.i.i.i6 = zext nneg i32 %18 to i64
  %add.i.i.i.i.i.i.i7 = add nuw nsw i64 %mul.i.i.i.i.i.i.i5, %conv.i2.i.i.i.i.i.i.i6
  %19 = call ptr addrspace(5) @llvm.amdgcn.implicit.offset()
  %20 = load i32, ptr addrspace(5) %19, align 4
  %zext.i.i.i.i.i.i.i.i8 = zext i32 %20 to i64
  %add4.i.i.i.i.i.i.i9 = add nuw nsw i64 %add.i.i.i.i.i.i.i7, %zext.i.i.i.i.i.i.i.i8
  %cmp.i.i.i10 = icmp ult i64 %add4.i.i.i.i.i.i.i9, 2147483648
  call void @llvm.assume(i1 %cmp.i.i.i10)
  %mul.i115.i = mul nuw nsw i64 %add4.i.i.i.i.i.i.i9, 3
  %v.i.i11 = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i107.i, i64 %mul.i115.i, i32 1
  %arrayidx.i.i.i12 = getelementptr inbounds i32, ptr addrspace(1) %v.i.i11, i64 1
  %21 = load i32, ptr addrspace(1) %arrayidx.i.i.i12, align 4, !tbaa !22
  %arrayidx.i118.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i110.i, i64 %mul.i115.i
  %22 = load i32, ptr addrspace(1) %arrayidx.i118.i, align 4, !tbaa !22
  %mul.i.i13 = mul nsw i32 %22, %21
  %add.i123.i = add nuw nsw i64 %mul.i115.i, 2
  %arrayidx.i124.i14 = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i113.i, i64 %add.i123.i
  %23 = load i8, ptr addrspace(1) %arrayidx.i124.i14, align 1, !tbaa !26
  %conv.i.i15 = sext i8 %23 to i32
  %add.i.i16 = add nsw i32 %mul.i.i13, %conv.i.i15
  %arrayidx.i128.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i.i1, i64 %mul.i115.i
  store i32 %add.i.i16, ptr addrspace(1) %arrayidx.i128.i, align 4, !tbaa !22
  %add.i133.i = add nuw nsw i64 %mul.i115.i, 1
  %v53.i.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i107.i, i64 %add.i133.i, i32 1
  %arrayidx.i.i136.i = getelementptr inbounds i32, ptr addrspace(1) %v53.i.i, i64 1
  %24 = load i32, ptr addrspace(1) %arrayidx.i.i136.i, align 4, !tbaa !22
  %arrayidx.i141.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i110.i, i64 %add.i133.i
  %25 = load i32, ptr addrspace(1) %arrayidx.i141.i, align 4, !tbaa !22
  %mul69.i.i = mul nsw i32 %25, %24
  %arrayidx.i147.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i113.i, i64 %add.i133.i
  %26 = load i8, ptr addrspace(1) %arrayidx.i147.i, align 1, !tbaa !26
  %conv84.i.i = sext i8 %26 to i32
  %add85.i.i = add nsw i32 %mul69.i.i, %conv84.i.i
  %arrayidx.i153.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i.i1, i64 %add.i133.i
  store i32 %add85.i.i, ptr addrspace(1) %arrayidx.i153.i, align 4, !tbaa !22
  %v114.i.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i107.i, i64 %add.i123.i, i32 1
  %arrayidx.i.i161.i = getelementptr inbounds i32, ptr addrspace(1) %v114.i.i, i64 1
  %27 = load i32, ptr addrspace(1) %arrayidx.i.i161.i, align 4, !tbaa !22
  %arrayidx.i166.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i110.i, i64 %add.i123.i
  %28 = load i32, ptr addrspace(1) %arrayidx.i166.i, align 4, !tbaa !22
  %mul130.i.i = mul nsw i32 %28, %27
  %arrayidx.i170.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i113.i, i64 %mul.i115.i
  %29 = load i8, ptr addrspace(1) %arrayidx.i170.i, align 1, !tbaa !26
  %conv140.i.i = sext i8 %29 to i32
  %add141.i.i = add nsw i32 %mul130.i.i, %conv140.i.i
  %arrayidx.i176.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i.i1, i64 %add.i123.i
  store i32 %add141.i.i, ptr addrspace(1) %arrayidx.i176.i, align 4, !tbaa !22
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nounwind speculatable memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #3 = { "frame-pointer"="all" "target-cpu"="gfx1031" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32" "uniform-work-group-size"="true" }

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!opencl.ocl.version = !{!2}
!llvm.ident = !{!3, !4}
!llvm.module.flags = !{!5, !6, !7, !8}
!sycl.specialization-constants = !{}
!sycl.specialization-constants-default-values = !{}
!amdgcn.annotations = !{!9}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{i32 2, i32 0}
!3 = !{!"clang version 18.0.0git (git@github.com:jopperm/llvm.git bfcbf22abcc1ed0986e22df57cf6e9fc2ab792af)"}
!4 = !{!"AMD clang version 16.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-5.6.1 23332 4f9bb99d78a4d8d9770be38b91ebd004ea4d2a3a)"}
!5 = !{i32 1, !"amdgpu_code_object_version", i32 400}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{i32 8, !"PIC Level", i32 1}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{ptr @fused_0, !"kernel", i32 1}
!10 = !{i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1}
!11 = !{i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false}
!12 = !{!"private", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"private", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!13 = !{i64 3, !"", !"", !"", !"", !"", !"", !"", i64 3, !"", !"", !"", !"", !"", !"", !"", !"", !"", !""}
!14 = !{i64 32, !"", !"", !"", !"", !"", !"", !"", i64 1, !"", !"", !"", !"", !"", !"", !"", !"", !"", !""}
!15 = !{!"", !"", !"", !"", !"", !"\80\01\00\00\00\00\00\00", !"\80\01\00\00\00\00\00\00", !"\00\00\00\00\00\00\00\00", !"", !"", !"", !"", !"", !"\80\01\00\00\00\00\00\00", !"\80\01\00\00\00\00\00\00", !"\00\00\00\00\00\00\00\00", !"\80\01\00\00\00\00\00\00", !"\80\01\00\00\00\00\00\00", !"\00\00\00\00\00\00\00\00"}
!16 = !{!17, !17, i64 0}
!17 = !{!"short", !18, i64 0}
!18 = !{!"omnipotent char", !19, i64 0}
!19 = !{!"Simple C/C++ TBAA"}
!20 = !{i32 0, i32 1024}
!21 = !{}
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!24, !24, i64 0}
