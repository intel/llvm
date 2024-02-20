; REQUIRES: hip_amd
; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext \
; RUN: -passes=sycl-internalization -S %s | FileCheck %s

; This test is the IR version of
; sycl/test-e2e/KernelFusion/internalize_non_unit_localsize.cpp for HIP.
; In contrast to the SPIR-V and CUDA versions, the sycl::vec in the test data
; structure is addressed via a multi-index GEP with a non-zero first index.

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8"
target triple = "amdgcn-amd-amdhsa"

%struct.MyStruct = type { i32, %"class.sycl::_V1::vec" }
%"class.sycl::_V1::vec" = type { <3 x i32> }

declare i32 @llvm.amdgcn.workgroup.id.x() #0
declare align 4 ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #0
declare i32 @llvm.amdgcn.workitem.id.x() #0
declare ptr addrspace(5) @llvm.amdgcn.implicit.offset() #1

define amdgpu_kernel void @fused_0(ptr addrspace(1) noundef align 16 %KernelOne__arg_accTmp31,
     i64 %KernelOne__arg_accTmp.coerce,
     i64 %KernelOne__arg_accTmp.coerce3,
     i64 %KernelOne__arg_accTmp.coerce7,
     ptr addrspace(1) noundef align 4 %KernelOne__arg_accIn35,
     i64 %KernelOne__arg_accIn.coerce,
     i64 %KernelOne__arg_accIn.coerce13,
     i64 %KernelOne__arg_accIn.coerce17,
     ptr addrspace(1) noundef align 1 %KernelOne__arg_accTmp239,
     i64 %KernelOne__arg_accTmp2.coerce,
     i64 %KernelOne__arg_accTmp2.coerce24,
     i64 %KernelOne__arg_accTmp2.coerce28,
     ptr addrspace(1) noundef align 4 %KernelTwo__arg_accOut30,
     i64 %KernelTwo__arg_accOut.coerce,
     i64 %KernelTwo__arg_accOut.coerce3,
     i64 %KernelTwo__arg_accOut.coerce7)
       #3 !sycl.kernel.promote !12 !sycl.kernel.promote.localsize !13 !sycl.kernel.promote.elemsize !14 {
; CHECK-LABEL: define amdgpu_kernel void @fused_0(
; CHECK-SAME: i64 [[KERNELONE__ARG_ACCTMP_COERCE7:%[^,]*accTmp.coerce7]]
; CHECK-SAME: i64 [[KERNELONE__ARG_ACCTMP2_COERCE28:%[^,]*accTmp2.coerce28]]
; CHECK:         entry:
; CHECK:           [[TMP0:%.*]] = alloca i8, i64 3, align 1, addrspace(5)
; CHECK:           [[TMP1:%.*]] = alloca i8, i64 96, align 16, addrspace(5)
; CHECK:           [[TMP2:%.*]] = urem i64 [[KERNELONE__ARG_ACCTMP_COERCE7]], 3
; CHECK:           [[TMP3:%.*]] = urem i64 [[KERNELONE__ARG_ACCTMP2_COERCE28]], 3
; CHECK:           [[MUL:%.*]] = mul nuw nsw i64 [[GLOBAL_ID:.*]], 3
; CHECK:           [[ADD:%.*]] = add nuw nsw i64 [[MUL]], 1
; CHECK:           [[TMP11:%.*]] = add i64 [[TMP2]], [[ADD]]
; CHECK:           [[TMP12:%.*]] = urem i64 [[TMP11]], 3

; COM:             This is a multi-index GEP into the aggregate. We have to remap the first index.
; CHECK:           [[V:%.*]] = getelementptr inbounds %struct.MyStruct, ptr addrspace(5) [[TMP1]], i64 [[TMP12]], i32 1

; COM:             This is a single-index GEP which shall not be remapped because its pointer operand already points into the struct (see above).
; CHECK:           [[ARRAYIDX_1:%.*]] = getelementptr inbounds i32, ptr addrspace(5) [[V]], i64 1

; CHECK:           store i32 {{.*}}, ptr addrspace(5) [[ARRAYIDX_1]], align 4
; CHECK:           [[TMP13:%.*]] = add i64 [[TMP3]], [[ADD]]
; CHECK:           [[TMP14:%.*]] = urem i64 [[TMP13]], 3

; COM:             This i8-GEP was remapped because it selects an element of the underlying i8-buffer
; CHECK:           [[ARRAYIDX_2:%.*]] = getelementptr inbounds i8, ptr addrspace(5) [[TMP0]], i64 [[TMP14]]
; CHECK:           store i8 {{.*}}, ptr addrspace(5) [[ARRAYIDX_2]], align 1
; CHECK:           store i32 {{.*}} ptr addrspace(1)
; CHECK:           ret void
;
entry:
  %add.ptr.j2 = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %KernelOne__arg_accTmp31, i64 %KernelOne__arg_accTmp.coerce7
  %add.ptr.i82.i = getelementptr inbounds i32, ptr addrspace(1) %KernelOne__arg_accIn35, i64 %KernelOne__arg_accIn.coerce17
  %add.ptr.i85.i = getelementptr inbounds i8, ptr addrspace(1) %KernelOne__arg_accTmp239, i64 %KernelOne__arg_accTmp2.coerce28
  %0 = call i32 @llvm.amdgcn.workgroup.id.x()
  %conv.i1.j7 = zext i32 %0 to i64
  %1 = call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %arrayidx.j8 = getelementptr inbounds i16, ptr addrspace(4) %1, i64 2
  %2 = load i16, ptr addrspace(4) %arrayidx.j8, align 4
  %conv.j8 = zext i16 %2 to i64
  %mul.j7 = mul nuw nsw i64 %conv.j8, %conv.i1.j7
  %3 = call i32 @llvm.amdgcn.workitem.id.x(), !range !20, !noundef !21
  %conv.i2.j7 = zext nneg i32 %3 to i64
  %add.j7 = add nuw nsw i64 %mul.j7, %conv.i2.j7
  %4 = call ptr addrspace(5) @llvm.amdgcn.implicit.offset()
  %5 = load i32, ptr addrspace(5) %4, align 4
  %zext.j8 = zext i32 %5 to i64
  %add4.j7 = add nuw nsw i64 %add.j7, %zext.j8
  %mul.j2 = mul nuw nsw i64 %add4.j7, 3
  %add.j2 = add nuw nsw i64 %mul.j2, 1
  %arrayidx.j2 = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i82.i, i64 %add.j2
  %6 = load i32, ptr addrspace(1) %arrayidx.j2, align 4
  %v.j2 = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.j2, i64 %add.j2, i32 1
  %arrayidx.j3 = getelementptr inbounds i32, ptr addrspace(1) %v.j2, i64 1
  store i32 %6, ptr addrspace(1) %arrayidx.j3, align 4
  %conv.j2 = trunc i32 %6 to i8
  %arrayidx.i104.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i85.i, i64 %add.j2
  store i8 %conv.j2, ptr addrspace(1) %arrayidx.i104.i, align 1
  %add.ptr.i.i1 = getelementptr inbounds i32, ptr addrspace(1) %KernelTwo__arg_accOut30, i64 %KernelTwo__arg_accOut.coerce7
  %add.ptr.i81.i = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %KernelOne__arg_accTmp31, i64 %KernelOne__arg_accTmp.coerce7
  %add.ptr.i84.i = getelementptr inbounds i8, ptr addrspace(1) %KernelOne__arg_accTmp239, i64 %KernelOne__arg_accTmp2.coerce28
  %7 = call i32 @llvm.amdgcn.workgroup.id.x()
  %conv.i1.j6.i2 = zext i32 %7 to i64
  %8 = call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %arrayidx.j7.i3 = getelementptr inbounds i16, ptr addrspace(4) %8, i64 2
  %9 = load i16, ptr addrspace(4) %arrayidx.j7.i3, align 4
  %conv.j7.i4 = zext i16 %9 to i64
  %mul.j6.i5 = mul nuw nsw i64 %conv.j7.i4, %conv.i1.j6.i2
  %10 = call i32 @llvm.amdgcn.workitem.id.x(), !range !20, !noundef !21
  %conv.i2.j6.i6 = zext nneg i32 %10 to i64
  %add.j6.i7 = add nuw nsw i64 %mul.j6.i5, %conv.i2.j6.i6
  %11 = call ptr addrspace(5) @llvm.amdgcn.implicit.offset()
  %12 = load i32, ptr addrspace(5) %11, align 4
  %zext.j7.i8 = zext i32 %12 to i64
  %add4.j6.i9 = add nuw nsw i64 %add.j6.i7, %zext.j7.i8
  %mul.i.i11 = mul nuw nsw i64 %add4.j6.i9, 3
  %add.i87.i = add nuw nsw i64 %mul.i.i11, 1
  %v.i.i12 = getelementptr inbounds %struct.MyStruct, ptr addrspace(1) %add.ptr.i81.i, i64 %add.i87.i, i32 1
  %arrayidx.j2.i13 = getelementptr inbounds i32, ptr addrspace(1) %v.i.i12, i64 1
  %13 = load i32, ptr addrspace(1) %arrayidx.j2.i13, align 4
  %arrayidx.i92.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i84.i, i64 %add.i87.i
  %14 = load i8, ptr addrspace(1) %arrayidx.i92.i, align 1
  %conv.i.i14 = sext i8 %14 to i32
  %add.i.i15 = add nsw i32 %13, %conv.i.i14
  %arrayidx.i98.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i.i1, i64 %add.i87.i
  store i32 %add.i.i15, ptr addrspace(1) %arrayidx.i98.i, align 4
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nounwind speculatable memory(none) }
attributes #3 = { "frame-pointer"="all" "target-cpu"="gfx1031" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx10-3-insts,+gfx10-insts,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize32" "uniform-work-group-size"="true" }

!amdgcn.annotations = !{!9}

!9 = !{ptr @fused_0, !"kernel", i32 1}
!12 = !{!"private", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"private", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!13 = !{i64 3, !"", !"", !"", !"", !"", !"", !"", i64 3, !"", !"", !"", !"", !"", !"", !""}
!14 = !{i64 32, !"", !"", !"", !"", !"", !"", !"", i64 1, !"", !"", !"", !"", !"", !"", !""}
!20 = !{i32 0, i32 1024}
!21 = !{}
