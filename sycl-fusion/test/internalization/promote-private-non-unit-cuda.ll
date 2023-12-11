; REQUIRES: cuda
; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext \
; RUN: -passes=sycl-internalization --sycl-info-path %S/../kernel-fusion/kernel-info.yaml -S %s | FileCheck %s

; This test is the IR version of
; sycl/test-e2e/KernelFusion/internalize_non_unit_localsize.cpp for CUDA

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }
%struct.MyStruct = type { i32, %"class.sycl::_V1::vec" }
%"class.sycl::_V1::vec" = type { <3 x i32> }

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: nounwind speculatable memory(none)
declare ptr @llvm.nvvm.implicit.offset() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef %0) #2

define void @fused_0(ptr addrspace(1) noundef align 16 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp3, ptr addrspace(1) noundef align 4 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn6, ptr addrspace(1) noundef align 1 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp27, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp210, ptr addrspace(1) noundef align 4 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut3, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn9) #3 !kernel_arg_buffer_location !11 !kernel_arg_runtime_aligned !12 !kernel_arg_exclusive_ptr !12 !sycl.kernel.promote !13 !sycl.kernel.promote.localsize !14 !sycl.kernel.promote.elemsize !15 !sycl.kernel.constants !16 {
; CHECK-LABEL: define void @fused_0(
; CHECK-SAME:      ptr noundef byval(%"class.sycl::_V1::range") align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP3:%.*]], ptr addrspace(1) noundef align 4 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCIN:%.*]], ptr noundef byval(%"class.sycl::_V1::range") align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCIN6:%.*]], ptr noundef byval(%"class.sycl::_V1::range") align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE_CLES2_E9KERNELONE__ARG_ACCTMP210:%.*]], ptr addrspace(1) noundef align 4 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE0_CLES2_E9KERNELTWO__ARG_ACCOUT:%.*]], ptr noundef byval(%"class.sycl::_V1::range") align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE0_CLES2_E9KERNELTWO__ARG_ACCOUT3:%.*]], ptr noundef byval(%"class.sycl::_V1::range") align 8 [[_ZTSZZ4MAINENKULRN4SYCL3_V17HANDLEREE0_CLES2_E9KERNELTWO__ARG_ACCIN9:%.*]]) #[[ATTR5:[0-9]+]] !kernel_arg_buffer_location !11 !kernel_arg_runtime_aligned !12 !kernel_arg_exclusive_ptr !12 !sycl.kernel.constants [[META13:![0-9]+]] {
; CHECK:         entry:
; CHECK:           [[TMP0:%.*]] = alloca i8, i64 3, align 1
; CHECK:           [[TMP1:%.*]] = alloca i8, i64 96, align 16
; CHECK:           [[TMP2:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ADD_PTR_I_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT:%.*]], ptr [[TMP1]], i64 [[TMP2]]
; CHECK:           [[TMP3:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ADD_PTR_I43_I:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP3]]
; CHECK:           [[TMP11:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I50_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr [[TMP1]], i64 [[TMP11]]
; CHECK:           [[ARRAYIDX_I_I_I:%.*]] = getelementptr inbounds i8, ptr [[ARRAYIDX_I50_I]], i64 20
; CHECK:           [[TMP14:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I67_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr [[TMP1]], i64 [[TMP14]]
; CHECK:           [[ARRAYIDX_I_I69_I:%.*]] = getelementptr inbounds i8, ptr [[ARRAYIDX_I67_I]], i64 20
; CHECK:           [[TMP17:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I86_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr [[TMP1]], i64 [[TMP17]]
; CHECK:           [[ARRAYIDX_I_I88_I:%.*]] = getelementptr inbounds i8, ptr [[ARRAYIDX_I86_I]], i64 20
; CHECK:           [[TMP21:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I102_I:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP21]]
; CHECK:           [[TMP25:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I120_I:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP25]]
; CHECK:           [[TMP29:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I135_I:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP29]]
; CHECK:           [[TMP30:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ADD_PTR_I45_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr [[TMP1]], i64 [[TMP30]]
; CHECK:           [[TMP31:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ADD_PTR_I57_I:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP31]]
; CHECK:           [[TMP38:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I_I17:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr [[TMP1]], i64 [[TMP38]]
; CHECK:           [[ARRAYIDX_I_I_I19:%.*]] = getelementptr inbounds i8, ptr [[ARRAYIDX_I_I17]], i64 20
; CHECK:           [[TMP42:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I74_I:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP42]]
; CHECK:           [[TMP45:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I89_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr [[TMP1]], i64 [[TMP45]]
; CHECK:           [[ARRAYIDX_I_I91_I:%.*]] = getelementptr inbounds i8, ptr [[ARRAYIDX_I89_I]], i64 20
; CHECK:           [[TMP49:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I108_I:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP49]]
; CHECK:           [[TMP52:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I126_I:%.*]] = getelementptr inbounds [[STRUCT_MYSTRUCT]], ptr [[TMP1]], i64 [[TMP52]]
; CHECK:           [[ARRAYIDX_I_I128_I:%.*]] = getelementptr inbounds i8, ptr [[ARRAYIDX_I126_I]], i64 20
; CHECK:           [[TMP56:%.*]] = urem i64 {{.*}}, 3
; CHECK:           [[ARRAYIDX_I142_I:%.*]] = getelementptr inbounds i8, ptr [[TMP0]], i64 [[TMP56]]
; CHECK:           ret void
;
entry:
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp2103.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp210, align 1
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn62.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn6, align 1
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp31.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp3, align 1
  %add.ptr.i.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp31.sroa.0.0.copyload
  %add.ptr.i37.i = getelementptr inbounds i32, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn62.sroa.0.0.copyload
  %add.ptr.i43.i = getelementptr inbounds i8, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp27, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp2103.sroa.0.0.copyload
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %conv.i1.i.i.i.i.i.i.i = sext i32 %0 to i64
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %conv.i3.i.i.i.i.i.i.i = sext i32 %1 to i64
  %mul.i.i.i.i.i.i.i = mul nsw i64 %conv.i3.i.i.i.i.i.i.i, %conv.i1.i.i.i.i.i.i.i
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %conv.i2.i.i.i.i.i.i.i = sext i32 %2 to i64
  %add.i.i.i.i.i.i.i = add nsw i64 %mul.i.i.i.i.i.i.i, %conv.i2.i.i.i.i.i.i.i
  %3 = call ptr @llvm.nvvm.implicit.offset()
  %4 = load i32, ptr %3, align 4, !tbaa !17
  %conv.i.i.i.i.i.i.i.i = zext i32 %4 to i64
  %add4.i.i.i.i.i.i.i = add nsw i64 %add.i.i.i.i.i.i.i, %conv.i.i.i.i.i.i.i.i
  %cmp.i.i.i.i = icmp ult i64 %add4.i.i.i.i.i.i.i, 2147483648
  call void @llvm.assume(i1 %cmp.i.i.i.i)
  %mul.i.i = mul nuw nsw i64 %add4.i.i.i.i.i.i.i, 3
  %arrayidx.i.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i37.i, i64 %mul.i.i
  %5 = load i32, ptr addrspace(1) %arrayidx.i.i, align 4, !tbaa !21
  %arrayidx.i50.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i.i, i64 %mul.i.i
  %arrayidx.i.i.i = getelementptr inbounds i8, ptr addrspace(1) %arrayidx.i50.i, i64 20
  store i32 %5, ptr addrspace(1) %arrayidx.i.i.i, align 4, !tbaa !21
  %add.i.i = add nuw nsw i64 %mul.i.i, 1
  %arrayidx.i58.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i37.i, i64 %add.i.i
  %6 = load i32, ptr addrspace(1) %arrayidx.i58.i, align 4, !tbaa !21
  %arrayidx.i67.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i.i, i64 %add.i.i
  %arrayidx.i.i69.i = getelementptr inbounds i8, ptr addrspace(1) %arrayidx.i67.i, i64 20
  store i32 %6, ptr addrspace(1) %arrayidx.i.i69.i, align 4, !tbaa !21
  %add.i74.i = add nuw nsw i64 %mul.i.i, 2
  %arrayidx.i77.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i37.i, i64 %add.i74.i
  %7 = load i32, ptr addrspace(1) %arrayidx.i77.i, align 4, !tbaa !21
  %arrayidx.i86.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i.i, i64 %add.i74.i
  %arrayidx.i.i88.i = getelementptr inbounds i8, ptr addrspace(1) %arrayidx.i86.i, i64 20
  store i32 %7, ptr addrspace(1) %arrayidx.i.i88.i, align 4, !tbaa !21
  %8 = load i32, ptr addrspace(1) %arrayidx.i.i, align 4, !tbaa !21
  %9 = trunc i32 %8 to i8
  %conv.i.i = xor i8 %9, -86
  %arrayidx.i102.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i43.i, i64 %add.i74.i
  store i8 %conv.i.i, ptr addrspace(1) %arrayidx.i102.i, align 1, !tbaa !25
  %10 = load i32, ptr addrspace(1) %arrayidx.i58.i, align 4, !tbaa !21
  %11 = trunc i32 %10 to i8
  %conv65.i.i = xor i8 %11, -86
  %arrayidx.i120.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i43.i, i64 %add.i.i
  store i8 %conv65.i.i, ptr addrspace(1) %arrayidx.i120.i, align 1, !tbaa !25
  %12 = load i32, ptr addrspace(1) %arrayidx.i77.i, align 4, !tbaa !21
  %13 = trunc i32 %12 to i8
  %conv83.i.i = xor i8 %13, -86
  %arrayidx.i135.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i43.i, i64 %mul.i.i
  store i8 %conv83.i.i, ptr addrspace(1) %arrayidx.i135.i, align 1, !tbaa !25
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp2107.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp210, align 1
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn96.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn9, align 1
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp35.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp3, align 1
  %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut34.sroa.0.0.copyload = load i64, ptr %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut3, align 1
  %add.ptr.i.i8 = getelementptr inbounds i32, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accOut34.sroa.0.0.copyload
  %add.ptr.i45.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp35.sroa.0.0.copyload
  %add.ptr.i51.i = getelementptr inbounds i32, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accIn, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo__arg_accIn96.sroa.0.0.copyload
  %add.ptr.i57.i = getelementptr inbounds i8, ptr addrspace(1) %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp27, i64 %_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne__arg_accTmp2107.sroa.0.0.copyload
  %14 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %conv.i1.i.i.i.i.i.i.i9 = sext i32 %14 to i64
  %15 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %conv.i3.i.i.i.i.i.i.i10 = sext i32 %15 to i64
  %mul.i.i.i.i.i.i.i11 = mul nsw i64 %conv.i3.i.i.i.i.i.i.i10, %conv.i1.i.i.i.i.i.i.i9
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %conv.i2.i.i.i.i.i.i.i12 = sext i32 %16 to i64
  %add.i.i.i.i.i.i.i13 = add nsw i64 %mul.i.i.i.i.i.i.i11, %conv.i2.i.i.i.i.i.i.i12
  %17 = call ptr @llvm.nvvm.implicit.offset()
  %18 = load i32, ptr %17, align 4, !tbaa !17
  %conv.i.i.i.i.i.i.i.i14 = zext i32 %18 to i64
  %add4.i.i.i.i.i.i.i15 = add nsw i64 %add.i.i.i.i.i.i.i13, %conv.i.i.i.i.i.i.i.i14
  %cmp.i.i.i.i16 = icmp ult i64 %add4.i.i.i.i.i.i.i15, 2147483648
  call void @llvm.assume(i1 %cmp.i.i.i.i16)
  %mul.i59.i = mul nuw nsw i64 %add4.i.i.i.i.i.i.i15, 3
  %arrayidx.i.i17 = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i45.i, i64 %mul.i59.i
  %arrayidx.i.i.i19 = getelementptr inbounds i8, ptr addrspace(1) %arrayidx.i.i17, i64 20
  %19 = load i32, ptr addrspace(1) %arrayidx.i.i.i19, align 4, !tbaa !21
  %arrayidx.i65.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i51.i, i64 %mul.i59.i
  %20 = load i32, ptr addrspace(1) %arrayidx.i65.i, align 4, !tbaa !21
  %mul.i.i20 = mul nsw i32 %20, %19
  %add.i71.i = add nuw nsw i64 %mul.i59.i, 2
  %arrayidx.i74.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i57.i, i64 %add.i71.i
  %21 = load i8, ptr addrspace(1) %arrayidx.i74.i, align 1, !tbaa !25
  %conv.i.i21 = sext i8 %21 to i32
  %add.i.i22 = add nsw i32 %mul.i.i20, %conv.i.i21
  %arrayidx.i80.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i.i8, i64 %mul.i59.i
  store i32 %add.i.i22, ptr addrspace(1) %arrayidx.i80.i, align 4, !tbaa !21
  %add.i86.i = add nuw nsw i64 %mul.i59.i, 1
  %arrayidx.i89.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i45.i, i64 %add.i86.i
  %arrayidx.i.i91.i = getelementptr inbounds i8, ptr addrspace(1) %arrayidx.i89.i, i64 20
  %22 = load i32, ptr addrspace(1) %arrayidx.i.i91.i, align 4, !tbaa !21
  %arrayidx.i99.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i51.i, i64 %add.i86.i
  %23 = load i32, ptr addrspace(1) %arrayidx.i99.i, align 4, !tbaa !21
  %mul37.i.i = mul nsw i32 %23, %22
  %arrayidx.i108.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i57.i, i64 %add.i86.i
  %24 = load i8, ptr addrspace(1) %arrayidx.i108.i, align 1, !tbaa !25
  %conv46.i.i = sext i8 %24 to i32
  %add47.i.i = add nsw i32 %mul37.i.i, %conv46.i.i
  %arrayidx.i117.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i.i8, i64 %add.i86.i
  store i32 %add47.i.i, ptr addrspace(1) %arrayidx.i117.i, align 4, !tbaa !21
  %arrayidx.i126.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i45.i, i64 %add.i71.i
  %arrayidx.i.i128.i = getelementptr inbounds i8, ptr addrspace(1) %arrayidx.i126.i, i64 20
  %25 = load i32, ptr addrspace(1) %arrayidx.i.i128.i, align 4, !tbaa !21
  %arrayidx.i136.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i51.i, i64 %add.i71.i
  %26 = load i32, ptr addrspace(1) %arrayidx.i136.i, align 4, !tbaa !21
  %mul74.i.i = mul nsw i32 %26, %25
  %arrayidx.i142.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i57.i, i64 %mul.i59.i
  %27 = load i8, ptr addrspace(1) %arrayidx.i142.i, align 1, !tbaa !25
  %conv80.i.i = sext i8 %27 to i32
  %add81.i.i = add nsw i32 %mul74.i.i, %conv80.i.i
  %arrayidx.i151.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i.i8, i64 %add.i71.i
  store i32 %add81.i.i, ptr addrspace(1) %arrayidx.i151.i, align 4, !tbaa !21
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly %0, ptr noalias nocapture readonly %1, i64 %2, i1 immarg %3) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg %0, ptr nocapture %1) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg %0, ptr nocapture %1) #5

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nounwind speculatable memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #3 = { "frame-pointer"="all" "target-cpu"="sm_80" "target-features"="+ptx81,+sm_80" "uniform-work-group-size"="true" }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!llvm.ident = !{!2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !3, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !3}
!nvvmir.version = !{!4, !4}
!llvm.module.flags = !{!5, !6, !7, !8, !9}
!sycl.specialization-constants = !{}
!sycl.specialization-constants-default-values = !{}
!nvvm.annotations = !{!10}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 18.0.0git (git@github.com:jopperm/llvm.git a22b0f1a212a8f29e803ffbde108bbffeebe4ee8)"}
!3 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!4 = !{i32 2, i32 0}
!5 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 1]}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{i32 7, !"nvvm-reflect-ftz", i32 0}
!8 = !{i32 7, !"nvvm-reflect-prec-sqrt", i32 0}
!9 = !{i32 7, !"frame-pointer", i32 2}
!10 = !{ptr @fused_0, !"kernel", i32 1}
!11 = !{i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1}
!12 = !{i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 false}
!13 = !{!"private", !"none", !"none", !"none", !"private", !"none", !"none", !"none", !"none"}
!14 = !{i64 3, !"", !"", !"", i64 3, !"", !"", !"", !""}
!15 = !{i64 32, !"", !"", !"", i64 1, !"", !"", !"", !""}
!16 = !{!"", !"", !"", !"\00\00\00\00\00\00\00\00", !"", !"", !"", !"\00\00\00\00\00\00\00\00", !"\00\00\00\00\00\00\00\00"}
!17 = !{!18, !18, i64 0}
!18 = !{!"int", !19, i64 0}
!19 = !{!"omnipotent char", !20, i64 0}
!20 = !{!"Simple C/C++ TBAA"}
!21 = !{!22, !22, i64 0}
!22 = !{!"int", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!23, !23, i64 0}
