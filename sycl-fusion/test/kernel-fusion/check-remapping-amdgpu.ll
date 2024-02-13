; REQUIRES: hip_amd
; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext \
; RUN:   -passes=sycl-kernel-fusion -S %s | FileCheck %s

; This tests checks that AMDGPU intrinsics are correctly remapped when fusing
; kernels with different ND-ranges.

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8"
target triple = "amdgcn-amd-amdhsa"

declare noundef i32 @llvm.amdgcn.workitem.id.x() #0
declare noundef i32 @llvm.amdgcn.workitem.id.y() #0
declare noundef i32 @llvm.amdgcn.workitem.id.z() #0
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #0
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #0
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #0
declare ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #0
declare ptr addrspace(5) @llvm.amdgcn.implicit.offset() #1

; Function Attrs: nounwind
define void @KernelOne(i32 %x) #2 !kernel_arg_addr_space !6 !kernel_arg_access_qual !6 !kernel_arg_type !6 !kernel_arg_type_qual !6 !kernel_arg_base_type !6 !kernel_arg_name !6 !work_group_size_hint !11 {
entry:
  %0 = call i32 @llvm.amdgcn.workitem.id.x() #0
  %1 = call i32 @llvm.amdgcn.workitem.id.y() #0
  %2 = call i32 @llvm.amdgcn.workitem.id.z() #0
  %3 = call i32 @llvm.amdgcn.workgroup.id.x() #0
  %4 = call i32 @llvm.amdgcn.workgroup.id.y() #0
  %5 = call i32 @llvm.amdgcn.workgroup.id.z() #0
  %6 = call ptr addrspace(5) @llvm.amdgcn.implicit.offset() #1
  ; Global size, x
  %DPtr = tail call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %arrayidx1 = getelementptr inbounds i32, ptr addrspace(4) %DPtr, i64 3
  %7 = load i32, ptr addrspace(4) %arrayidx1, align 4
  ; Global size, y
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(4) %DPtr, i64 4
  %8 = load i32, ptr addrspace(4) %arrayidx2, align 4
  ; Global size, z
  %arrayidx3 = getelementptr inbounds i32, ptr addrspace(4) %DPtr, i64 5
  %9 = load i32, ptr addrspace(4) %arrayidx3, align 4
  ; Workgroup size, x
  %arrayidx4 = getelementptr inbounds i16, ptr addrspace(4) %DPtr, i64 2
  %l4 = load i16, ptr addrspace(4) %arrayidx4, align 4
  %10 = zext i16 %l4 to i32
  ; Workgroup size, y
  %arrayidx5 = getelementptr inbounds i16, ptr addrspace(4) %DPtr, i64 3
  %l5 = load i16, ptr addrspace(4) %arrayidx5, align 4
  %11 = zext i16 %l5 to i32
  ; Workgroup size, z
  %arrayidx6 = getelementptr inbounds i16, ptr addrspace(4) %DPtr, i64 4
  %l6 = load i16, ptr addrspace(4) %arrayidx6, align 4
  %12 = zext i16 %l6 to i32
  ret void
}

declare !sycl.kernel.fused !13 !sycl.kernel.nd-ranges !15 !sycl.kernel.nd-range !24 void @fused_kernel()

; CHECK: @[[__GLOBAL_OFFSET_REMAPPER_3_2_3_7_2_1_1_3_48_1_1_2_1_1_X__CONST:[a-zA-Z0-9_$"\\.-]+]] = internal addrspace(5) constant [3 x i32] zeroinitializer
; CHECK: @[[__GLOBAL_OFFSET_REMAPPER_2_2_4_1_2_1_1_3_48_1_1_2_1_1_X__CONST:[a-zA-Z0-9_$"\\.-]+]] = internal addrspace(5) constant [3 x i32] zeroinitializer
; CHECK: @[[__GLOBAL_OFFSET_REMAPPER_1_48_1_1_2_1_1_3_48_1_1_2_1_1_X__CONST:[a-zA-Z0-9_$"\\.-]+]] = internal addrspace(5) constant [3 x i32] zeroinitializer
; CHECK: @[[__GLOBAL_OFFSET_REMAPPER_1_20_1_1_10_1_1_3_48_1_1_2_1_1_X__CONST:[a-zA-Z0-9_$"\\.-]+]] = internal addrspace(5) constant [3 x i32] zeroinitializer
; CHECK: @[[__GLOBAL_OFFSET_REMAPPER_1_10_1_1_10_1_1_3_48_1_1_2_1_1_X__CONST:[a-zA-Z0-9_$"\\.-]+]] = internal addrspace(5) constant [3 x i32] zeroinitializer

; CHECK-LABEL: define amdgpu_kernel void @fused_0(
; CHECK-SAME: i32 [[KERNELONE_X:%.*]], i32 [[KERNELONE_X1:%.*]], i32 [[KERNELONE_X2:%.*]]) !work_group_size_hint !3 !kernel_arg_buffer_location !4 !kernel_arg_runtime_aligned !4 !kernel_arg_exclusive_ptr !4 {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS:.*]]
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ult i32 [[TMP0]], 42
; CHECK-NEXT:    br i1 [[TMP1]], label [[TMP2:%.*]], label [[TMP25:%.*]]
; CHECK:       2:
; CHECK-NEXT:    [[TMP3:%.*]] = call i32 @__local_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP4:%.*]] = call i32 @__local_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP5:%.*]] = call i32 @__local_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP6:%.*]] = call i32 @__group_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP7:%.*]] = call i32 @__group_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP8:%.*]] = call i32 @__group_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP9:%.*]] = call ptr addrspace(5) @__global_offset_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS_GOFF:.*]]
; CHECK-NEXT:    [[DPTR_I:%.*]] = call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; CHECK-NEXT:    [[ARRAYIDX1_I:%.*]] = getelementptr inbounds i32, ptr addrspace(4) [[DPTR_I]], i64 3
; CHECK-NEXT:    [[TMP10:%.*]] = call i32 @__global_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP11:%.*]] = load i32, ptr addrspace(4) [[ARRAYIDX1_I]], align 4
; CHECK-NEXT:    [[ARRAYIDX2_I:%.*]] = getelementptr inbounds i32, ptr addrspace(4) [[DPTR_I]], i64 4
; CHECK-NEXT:    [[TMP12:%.*]] = call i32 @__global_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP13:%.*]] = load i32, ptr addrspace(4) [[ARRAYIDX2_I]], align 4
; CHECK-NEXT:    [[ARRAYIDX3_I:%.*]] = getelementptr inbounds i32, ptr addrspace(4) [[DPTR_I]], i64 5
; CHECK-NEXT:    [[TMP14:%.*]] = call i32 @__global_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP15:%.*]] = load i32, ptr addrspace(4) [[ARRAYIDX3_I]], align 4
; CHECK-NEXT:    [[ARRAYIDX4_I:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[DPTR_I]], i64 2
; CHECK-NEXT:    [[TMP16:%.*]] = call i32 @__local_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP17:%.*]] = trunc i32 [[TMP16]] to i16
; CHECK-NEXT:    [[L4_I:%.*]] = load i16, ptr addrspace(4) [[ARRAYIDX4_I]], align 4
; CHECK-NEXT:    [[TMP18:%.*]] = zext i16 [[TMP17]] to i32
; CHECK-NEXT:    [[ARRAYIDX5_I:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[DPTR_I]], i64 3
; CHECK-NEXT:    [[TMP19:%.*]] = call i32 @__local_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP20:%.*]] = trunc i32 [[TMP19]] to i16
; CHECK-NEXT:    [[L5_I:%.*]] = load i16, ptr addrspace(4) [[ARRAYIDX5_I]], align 4
; CHECK-NEXT:    [[TMP21:%.*]] = zext i16 [[TMP20]] to i32
; CHECK-NEXT:    [[ARRAYIDX6_I:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[DPTR_I]], i64 4
; CHECK-NEXT:    [[TMP22:%.*]] = call i32 @__local_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP23:%.*]] = trunc i32 [[TMP22]] to i16
; CHECK-NEXT:    [[L6_I:%.*]] = load i16, ptr addrspace(4) [[ARRAYIDX6_I]], align 4
; CHECK-NEXT:    [[TMP24:%.*]] = zext i16 [[TMP23]] to i32
; CHECK-NEXT:    br label [[TMP25]]
; CHECK:       25:
; CHECK-NEXT:    fence syncscope("workgroup") seq_cst
; CHECK-NEXT:    call void @llvm.amdgcn.s.barrier()
; CHECK-NEXT:    [[TMP26:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP27:%.*]] = icmp ult i32 [[TMP26]], 8
; CHECK-NEXT:    br i1 [[TMP27]], label [[TMP28:%.*]], label [[TMP51:%.*]]
; CHECK:       28:
; CHECK-NEXT:    [[TMP29:%.*]] = call i32 @__local_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP30:%.*]] = call i32 @__local_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP31:%.*]] = call i32 @__local_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP32:%.*]] = call i32 @__group_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP33:%.*]] = call i32 @__group_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP34:%.*]] = call i32 @__group_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP35:%.*]] = call ptr addrspace(5) @__global_offset_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS_GOFF]]
; CHECK-NEXT:    [[DPTR_I3:%.*]] = call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; CHECK-NEXT:    [[ARRAYIDX1_I4:%.*]] = getelementptr inbounds i32, ptr addrspace(4) [[DPTR_I3]], i64 3
; CHECK-NEXT:    [[TMP36:%.*]] = call i32 @__global_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP37:%.*]] = load i32, ptr addrspace(4) [[ARRAYIDX1_I4]], align 4
; CHECK-NEXT:    [[ARRAYIDX2_I5:%.*]] = getelementptr inbounds i32, ptr addrspace(4) [[DPTR_I3]], i64 4
; CHECK-NEXT:    [[TMP38:%.*]] = call i32 @__global_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP39:%.*]] = load i32, ptr addrspace(4) [[ARRAYIDX2_I5]], align 4
; CHECK-NEXT:    [[ARRAYIDX3_I6:%.*]] = getelementptr inbounds i32, ptr addrspace(4) [[DPTR_I3]], i64 5
; CHECK-NEXT:    [[TMP40:%.*]] = call i32 @__global_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP41:%.*]] = load i32, ptr addrspace(4) [[ARRAYIDX3_I6]], align 4
; CHECK-NEXT:    [[ARRAYIDX4_I7:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[DPTR_I3]], i64 2
; CHECK-NEXT:    [[TMP42:%.*]] = call i32 @__local_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP43:%.*]] = trunc i32 [[TMP42]] to i16
; CHECK-NEXT:    [[L4_I8:%.*]] = load i16, ptr addrspace(4) [[ARRAYIDX4_I7]], align 4
; CHECK-NEXT:    [[TMP44:%.*]] = zext i16 [[TMP43]] to i32
; CHECK-NEXT:    [[ARRAYIDX5_I9:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[DPTR_I3]], i64 3
; CHECK-NEXT:    [[TMP45:%.*]] = call i32 @__local_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP46:%.*]] = trunc i32 [[TMP45]] to i16
; CHECK-NEXT:    [[L5_I10:%.*]] = load i16, ptr addrspace(4) [[ARRAYIDX5_I9]], align 4
; CHECK-NEXT:    [[TMP47:%.*]] = zext i16 [[TMP46]] to i32
; CHECK-NEXT:    [[ARRAYIDX6_I11:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[DPTR_I3]], i64 4
; CHECK-NEXT:    [[TMP48:%.*]] = call i32 @__local_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP49:%.*]] = trunc i32 [[TMP48]] to i16
; CHECK-NEXT:    [[L6_I12:%.*]] = load i16, ptr addrspace(4) [[ARRAYIDX6_I11]], align 4
; CHECK-NEXT:    [[TMP50:%.*]] = zext i16 [[TMP49]] to i32
; CHECK-NEXT:    br label [[TMP51]]
; CHECK:       51:
; CHECK-NEXT:    fence syncscope("workgroup") seq_cst
; CHECK-NEXT:    call void @llvm.amdgcn.s.barrier()
; CHECK-NEXT:    [[TMP52:%.*]] = call i32 @__local_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP53:%.*]] = call i32 @__local_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP54:%.*]] = call i32 @__local_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP55:%.*]] = call i32 @__group_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP56:%.*]] = call i32 @__group_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP57:%.*]] = call i32 @__group_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP58:%.*]] = call ptr addrspace(5) @__global_offset_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS_GOFF]]
; CHECK-NEXT:    [[DPTR_I13:%.*]] = call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; CHECK-NEXT:    [[ARRAYIDX1_I14:%.*]] = getelementptr inbounds i32, ptr addrspace(4) [[DPTR_I13]], i64 3
; CHECK-NEXT:    [[TMP59:%.*]] = call i32 @__global_size_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP60:%.*]] = load i32, ptr addrspace(4) [[ARRAYIDX1_I14]], align 4
; CHECK-NEXT:    [[ARRAYIDX2_I15:%.*]] = getelementptr inbounds i32, ptr addrspace(4) [[DPTR_I13]], i64 4
; CHECK-NEXT:    [[TMP61:%.*]] = call i32 @__global_size_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP62:%.*]] = load i32, ptr addrspace(4) [[ARRAYIDX2_I15]], align 4
; CHECK-NEXT:    [[ARRAYIDX3_I16:%.*]] = getelementptr inbounds i32, ptr addrspace(4) [[DPTR_I13]], i64 5
; CHECK-NEXT:    [[TMP63:%.*]] = call i32 @__global_size_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP64:%.*]] = load i32, ptr addrspace(4) [[ARRAYIDX3_I16]], align 4
; CHECK-NEXT:    [[ARRAYIDX4_I17:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[DPTR_I13]], i64 2
; CHECK-NEXT:    [[TMP65:%.*]] = call i32 @__local_size_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP66:%.*]] = trunc i32 [[TMP65]] to i16
; CHECK-NEXT:    [[L4_I18:%.*]] = load i16, ptr addrspace(4) [[ARRAYIDX4_I17]], align 4
; CHECK-NEXT:    [[TMP67:%.*]] = zext i16 [[TMP66]] to i32
; CHECK-NEXT:    [[ARRAYIDX5_I19:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[DPTR_I13]], i64 3
; CHECK-NEXT:    [[TMP68:%.*]] = call i32 @__local_size_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP69:%.*]] = trunc i32 [[TMP68]] to i16
; CHECK-NEXT:    [[L5_I20:%.*]] = load i16, ptr addrspace(4) [[ARRAYIDX5_I19]], align 4
; CHECK-NEXT:    [[TMP70:%.*]] = zext i16 [[TMP69]] to i32
; CHECK-NEXT:    [[ARRAYIDX6_I21:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[DPTR_I13]], i64 4
; CHECK-NEXT:    [[TMP71:%.*]] = call i32 @__local_size_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP72:%.*]] = trunc i32 [[TMP71]] to i16
; CHECK-NEXT:    [[L6_I22:%.*]] = load i16, ptr addrspace(4) [[ARRAYIDX6_I21]], align 4
; CHECK-NEXT:    [[TMP73:%.*]] = zext i16 [[TMP72]] to i32
; CHECK-NEXT:    ret void

; CHECK-LABEL: define internal i32 @__global_linear_id_3_48_1_1_2_1_1(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_id_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = mul i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = call i32 @__global_id_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP3:%.*]] = mul i32 [[TMP2]], 1
; CHECK-NEXT:    [[TMP4:%.*]] = add i32 [[TMP3]], [[TMP1]]
; CHECK-NEXT:    [[TMP5:%.*]] = call i32 @__global_id_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP6:%.*]] = add i32 [[TMP5]], [[TMP4]]
; CHECK-NEXT:    ret i32 [[TMP6]]

; CHECK-LABEL: define internal i32 @__global_id_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @llvm.amdgcn.workgroup.id.z()
; CHECK-NEXT:    [[TMP1:%.*]] = call i32 @llvm.amdgcn.workitem.id.z()
; CHECK-NEXT:    [[TMP2:%.*]] = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[TMP2]], i64 4
; CHECK-NEXT:    [[TMP4:%.*]] = load i16, ptr addrspace(4) [[TMP3]], align 2
; CHECK-NEXT:    [[TMP5:%.*]] = zext i16 [[TMP4]] to i32
; CHECK-NEXT:    [[TMP6:%.*]] = mul i32 [[TMP0]], [[TMP5]]
; CHECK-NEXT:    [[TMP7:%.*]] = add i32 [[TMP6]], [[TMP1]]
; CHECK-NEXT:    ret i32 [[TMP7]]

; CHECK-LABEL: define internal i32 @__global_id_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @llvm.amdgcn.workgroup.id.y()
; CHECK-NEXT:    [[TMP1:%.*]] = call i32 @llvm.amdgcn.workitem.id.y()
; CHECK-NEXT:    [[TMP2:%.*]] = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[TMP2]], i64 3
; CHECK-NEXT:    [[TMP4:%.*]] = load i16, ptr addrspace(4) [[TMP3]], align 2
; CHECK-NEXT:    [[TMP5:%.*]] = zext i16 [[TMP4]] to i32
; CHECK-NEXT:    [[TMP6:%.*]] = mul i32 [[TMP0]], [[TMP5]]
; CHECK-NEXT:    [[TMP7:%.*]] = add i32 [[TMP6]], [[TMP1]]
; CHECK-NEXT:    ret i32 [[TMP7]]

; CHECK-LABEL: define internal i32 @__global_id_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @llvm.amdgcn.workgroup.id.x()
; CHECK-NEXT:    [[TMP1:%.*]] = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK-NEXT:    [[TMP2:%.*]] = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[TMP2]], i64 2
; CHECK-NEXT:    [[TMP4:%.*]] = load i16, ptr addrspace(4) [[TMP3]], align 2
; CHECK-NEXT:    [[TMP5:%.*]] = zext i16 [[TMP4]] to i32
; CHECK-NEXT:    [[TMP6:%.*]] = mul i32 [[TMP0]], [[TMP5]]
; CHECK-NEXT:    [[TMP7:%.*]] = add i32 [[TMP6]], [[TMP1]]
; CHECK-NEXT:    ret i32 [[TMP7]]

; CHECK-LABEL: define internal i32 @__local_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 6
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 1
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal i32 @__local_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 2
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 3
; CHECK-NEXT:    [[TMP3:%.*]] = urem i32 [[TMP2]], 1
; CHECK-NEXT:    ret i32 [[TMP3]]

; CHECK-LABEL: define internal i32 @__local_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = urem i32 [[TMP0]], 2
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 2
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal i32 @__group_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 6
; CHECK-NEXT:    [[TMP2:%.*]] = udiv i32 [[TMP1]], 1
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal i32 @__group_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 2
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 3
; CHECK-NEXT:    [[TMP3:%.*]] = udiv i32 [[TMP2]], 1
; CHECK-NEXT:    ret i32 [[TMP3]]

; CHECK-LABEL: define internal i32 @__group_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = urem i32 [[TMP0]], 2
; CHECK-NEXT:    [[TMP2:%.*]] = udiv i32 [[TMP1]], 2
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal ptr addrspace(5) @__global_offset_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS_GOFF]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret ptr addrspace(5) @__global_offset_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x__const

; CHECK-LABEL: define internal i32 @__global_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 7

; CHECK-LABEL: define internal i32 @__global_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 3

; CHECK-LABEL: define internal i32 @__global_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 2

; CHECK-LABEL: define internal i32 @__local_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__local_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__local_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 2

; CHECK-LABEL: define internal i32 @__local_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 2
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 1
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal i32 @__local_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 2
; CHECK-NEXT:    [[TMP3:%.*]] = urem i32 [[TMP2]], 2
; CHECK-NEXT:    ret i32 [[TMP3]]

; CHECK-LABEL: define internal i32 @__local_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = urem i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 1
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal i32 @__group_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 2
; CHECK-NEXT:    [[TMP2:%.*]] = udiv i32 [[TMP1]], 1
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal i32 @__group_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 2
; CHECK-NEXT:    [[TMP3:%.*]] = udiv i32 [[TMP2]], 2
; CHECK-NEXT:    ret i32 [[TMP3]]

; CHECK-LABEL: define internal i32 @__group_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = urem i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = udiv i32 [[TMP1]], 1
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal ptr addrspace(5) @__global_offset_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS_GOFF]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret ptr addrspace(5) @__global_offset_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x__const

; CHECK-LABEL: define internal i32 @__global_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 4

; CHECK-LABEL: define internal i32 @__global_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 2

; CHECK-LABEL: define internal i32 @__global_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__local_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__local_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 2

; CHECK-LABEL: define internal i32 @__local_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__local_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 2
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal i32 @__local_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 1
; CHECK-NEXT:    [[TMP3:%.*]] = urem i32 [[TMP2]], 1
; CHECK-NEXT:    ret i32 [[TMP3]]

; CHECK-LABEL: define internal i32 @__local_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = urem i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 1
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal i32 @__group_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = udiv i32 [[TMP1]], 2
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal i32 @__group_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 1
; CHECK-NEXT:    [[TMP3:%.*]] = udiv i32 [[TMP2]], 1
; CHECK-NEXT:    ret i32 [[TMP3]]

; CHECK-LABEL: define internal i32 @__group_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = urem i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = udiv i32 [[TMP1]], 1
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal ptr addrspace(5) @__global_offset_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS_GOFF]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret ptr addrspace(5) @__global_offset_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_x__const

; CHECK-LABEL: define internal i32 @__global_size_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 48

; CHECK-LABEL: define internal i32 @__global_size_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__global_size_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__local_size_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 2

; CHECK-LABEL: define internal i32 @__local_size_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__local_size_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

declare !sycl.kernel.fused !31 !sycl.kernel.nd-ranges !25 !sycl.kernel.nd-range !24 void @fused_kernel_1D()

; CHECK-LABEL: define amdgpu_kernel void @fused_1(
; CHECK-SAME: i32 [[KERNELONE_X:%.*]], i32 [[KERNELONE_X1:%.*]], i32 [[KERNELONE_X2:%.*]]) !work_group_size_hint !3 !kernel_arg_buffer_location !4 !kernel_arg_runtime_aligned !4 !kernel_arg_exclusive_ptr !4 {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ult i32 [[TMP0]], 20
; CHECK-NEXT:    br i1 [[TMP1]], label [[TMP2:%.*]], label [[TMP25:%.*]]
; CHECK:       2:
; CHECK-NEXT:    [[TMP3:%.*]] = call i32 @__local_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP4:%.*]] = call i32 @__local_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP5:%.*]] = call i32 @__local_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP6:%.*]] = call i32 @__group_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP7:%.*]] = call i32 @__group_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP8:%.*]] = call i32 @__group_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP9:%.*]] = call ptr addrspace(5) @__global_offset_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS_GOFF]]
; CHECK-NEXT:    [[DPTR_I:%.*]] = call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; CHECK-NEXT:    [[ARRAYIDX1_I:%.*]] = getelementptr inbounds i32, ptr addrspace(4) [[DPTR_I]], i64 3
; CHECK-NEXT:    [[TMP10:%.*]] = call i32 @__global_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP11:%.*]] = load i32, ptr addrspace(4) [[ARRAYIDX1_I]], align 4
; CHECK-NEXT:    [[ARRAYIDX2_I:%.*]] = getelementptr inbounds i32, ptr addrspace(4) [[DPTR_I]], i64 4
; CHECK-NEXT:    [[TMP12:%.*]] = call i32 @__global_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP13:%.*]] = load i32, ptr addrspace(4) [[ARRAYIDX2_I]], align 4
; CHECK-NEXT:    [[ARRAYIDX3_I:%.*]] = getelementptr inbounds i32, ptr addrspace(4) [[DPTR_I]], i64 5
; CHECK-NEXT:    [[TMP14:%.*]] = call i32 @__global_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP15:%.*]] = load i32, ptr addrspace(4) [[ARRAYIDX3_I]], align 4
; CHECK-NEXT:    [[ARRAYIDX4_I:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[DPTR_I]], i64 2
; CHECK-NEXT:    [[TMP16:%.*]] = call i32 @__local_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP17:%.*]] = trunc i32 [[TMP16]] to i16
; CHECK-NEXT:    [[L4_I:%.*]] = load i16, ptr addrspace(4) [[ARRAYIDX4_I]], align 4
; CHECK-NEXT:    [[TMP18:%.*]] = zext i16 [[TMP17]] to i32
; CHECK-NEXT:    [[ARRAYIDX5_I:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[DPTR_I]], i64 3
; CHECK-NEXT:    [[TMP19:%.*]] = call i32 @__local_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP20:%.*]] = trunc i32 [[TMP19]] to i16
; CHECK-NEXT:    [[L5_I:%.*]] = load i16, ptr addrspace(4) [[ARRAYIDX5_I]], align 4
; CHECK-NEXT:    [[TMP21:%.*]] = zext i16 [[TMP20]] to i32
; CHECK-NEXT:    [[ARRAYIDX6_I:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[DPTR_I]], i64 4
; CHECK-NEXT:    [[TMP22:%.*]] = call i32 @__local_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP23:%.*]] = trunc i32 [[TMP22]] to i16
; CHECK-NEXT:    [[L6_I:%.*]] = load i16, ptr addrspace(4) [[ARRAYIDX6_I]], align 4
; CHECK-NEXT:    [[TMP24:%.*]] = zext i16 [[TMP23]] to i32
; CHECK-NEXT:    br label [[TMP25]]
; CHECK:       25:
; CHECK-NEXT:    fence syncscope("workgroup") seq_cst
; CHECK-NEXT:    call void @llvm.amdgcn.s.barrier()
; CHECK-NEXT:    [[TMP26:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP27:%.*]] = icmp ult i32 [[TMP26]], 10
; CHECK-NEXT:    br i1 [[TMP27]], label [[TMP28:%.*]], label [[TMP51:%.*]]
; CHECK:       28:
; CHECK-NEXT:    [[TMP29:%.*]] = call i32 @__local_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP30:%.*]] = call i32 @__local_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP31:%.*]] = call i32 @__local_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP32:%.*]] = call i32 @__group_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP33:%.*]] = call i32 @__group_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP34:%.*]] = call i32 @__group_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP35:%.*]] = call ptr addrspace(5) @__global_offset_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS_GOFF]]
; CHECK-NEXT:    [[DPTR_I3:%.*]] = call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; CHECK-NEXT:    [[ARRAYIDX1_I4:%.*]] = getelementptr inbounds i32, ptr addrspace(4) [[DPTR_I3]], i64 3
; CHECK-NEXT:    [[TMP36:%.*]] = call i32 @__global_size_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP37:%.*]] = load i32, ptr addrspace(4) [[ARRAYIDX1_I4]], align 4
; CHECK-NEXT:    [[ARRAYIDX2_I5:%.*]] = getelementptr inbounds i32, ptr addrspace(4) [[DPTR_I3]], i64 4
; CHECK-NEXT:    [[TMP38:%.*]] = call i32 @__global_size_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP39:%.*]] = load i32, ptr addrspace(4) [[ARRAYIDX2_I5]], align 4
; CHECK-NEXT:    [[ARRAYIDX3_I6:%.*]] = getelementptr inbounds i32, ptr addrspace(4) [[DPTR_I3]], i64 5
; CHECK-NEXT:    [[TMP40:%.*]] = call i32 @__global_size_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP41:%.*]] = load i32, ptr addrspace(4) [[ARRAYIDX3_I6]], align 4
; CHECK-NEXT:    [[ARRAYIDX4_I7:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[DPTR_I3]], i64 2
; CHECK-NEXT:    [[TMP42:%.*]] = call i32 @__local_size_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP43:%.*]] = trunc i32 [[TMP42]] to i16
; CHECK-NEXT:    [[L4_I8:%.*]] = load i16, ptr addrspace(4) [[ARRAYIDX4_I7]], align 4
; CHECK-NEXT:    [[TMP44:%.*]] = zext i16 [[TMP43]] to i32
; CHECK-NEXT:    [[ARRAYIDX5_I9:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[DPTR_I3]], i64 3
; CHECK-NEXT:    [[TMP45:%.*]] = call i32 @__local_size_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP46:%.*]] = trunc i32 [[TMP45]] to i16
; CHECK-NEXT:    [[L5_I10:%.*]] = load i16, ptr addrspace(4) [[ARRAYIDX5_I9]], align 4
; CHECK-NEXT:    [[TMP47:%.*]] = zext i16 [[TMP46]] to i32
; CHECK-NEXT:    [[ARRAYIDX6_I11:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[DPTR_I3]], i64 4
; CHECK-NEXT:    [[TMP48:%.*]] = call i32 @__local_size_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP49:%.*]] = trunc i32 [[TMP48]] to i16
; CHECK-NEXT:    [[L6_I12:%.*]] = load i16, ptr addrspace(4) [[ARRAYIDX6_I11]], align 4
; CHECK-NEXT:    [[TMP50:%.*]] = zext i16 [[TMP49]] to i32
; CHECK-NEXT:    br label [[TMP51]]
; CHECK:       51:
; CHECK-NEXT:    fence syncscope("workgroup") seq_cst
; CHECK-NEXT:    call void @llvm.amdgcn.s.barrier()
; CHECK-NEXT:    [[TMP52:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP53:%.*]] = icmp ult i32 [[TMP52]], 20
; CHECK-NEXT:    br i1 [[TMP53]], label [[TMP54:%.*]], label [[TMP77:%.*]]
; CHECK:       54:
; CHECK-NEXT:    [[TMP55:%.*]] = call i32 @__local_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP56:%.*]] = call i32 @__local_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP57:%.*]] = call i32 @__local_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP58:%.*]] = call i32 @__group_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP59:%.*]] = call i32 @__group_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP60:%.*]] = call i32 @__group_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP61:%.*]] = call ptr addrspace(5) @__global_offset_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS_GOFF]]
; CHECK-NEXT:    [[DPTR_I13:%.*]] = call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
; CHECK-NEXT:    [[ARRAYIDX1_I14:%.*]] = getelementptr inbounds i32, ptr addrspace(4) [[DPTR_I13]], i64 3
; CHECK-NEXT:    [[TMP62:%.*]] = call i32 @__global_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP63:%.*]] = load i32, ptr addrspace(4) [[ARRAYIDX1_I14]], align 4
; CHECK-NEXT:    [[ARRAYIDX2_I15:%.*]] = getelementptr inbounds i32, ptr addrspace(4) [[DPTR_I13]], i64 4
; CHECK-NEXT:    [[TMP64:%.*]] = call i32 @__global_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP65:%.*]] = load i32, ptr addrspace(4) [[ARRAYIDX2_I15]], align 4
; CHECK-NEXT:    [[ARRAYIDX3_I16:%.*]] = getelementptr inbounds i32, ptr addrspace(4) [[DPTR_I13]], i64 5
; CHECK-NEXT:    [[TMP66:%.*]] = call i32 @__global_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP67:%.*]] = load i32, ptr addrspace(4) [[ARRAYIDX3_I16]], align 4
; CHECK-NEXT:    [[ARRAYIDX4_I17:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[DPTR_I13]], i64 2
; CHECK-NEXT:    [[TMP68:%.*]] = call i32 @__local_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP69:%.*]] = trunc i32 [[TMP68]] to i16
; CHECK-NEXT:    [[L4_I18:%.*]] = load i16, ptr addrspace(4) [[ARRAYIDX4_I17]], align 4
; CHECK-NEXT:    [[TMP70:%.*]] = zext i16 [[TMP69]] to i32
; CHECK-NEXT:    [[ARRAYIDX5_I19:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[DPTR_I13]], i64 3
; CHECK-NEXT:    [[TMP71:%.*]] = call i32 @__local_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP72:%.*]] = trunc i32 [[TMP71]] to i16
; CHECK-NEXT:    [[L5_I20:%.*]] = load i16, ptr addrspace(4) [[ARRAYIDX5_I19]], align 4
; CHECK-NEXT:    [[TMP73:%.*]] = zext i16 [[TMP72]] to i32
; CHECK-NEXT:    [[ARRAYIDX6_I21:%.*]] = getelementptr inbounds i16, ptr addrspace(4) [[DPTR_I13]], i64 4
; CHECK-NEXT:    [[TMP74:%.*]] = call i32 @__local_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP75:%.*]] = trunc i32 [[TMP74]] to i16
; CHECK-NEXT:    [[L6_I22:%.*]] = load i16, ptr addrspace(4) [[ARRAYIDX6_I21]], align 4
; CHECK-NEXT:    [[TMP76:%.*]] = zext i16 [[TMP75]] to i32
; CHECK-NEXT:    br label [[TMP77]]
; CHECK:       77:
; CHECK-NEXT:    ret void

; CHECK-LABEL: define internal i32 @__local_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 10
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal i32 @__local_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 1
; CHECK-NEXT:    [[TMP3:%.*]] = urem i32 [[TMP2]], 1
; CHECK-NEXT:    ret i32 [[TMP3]]

; CHECK-LABEL: define internal i32 @__local_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = urem i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 1
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal i32 @__group_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = udiv i32 [[TMP1]], 10
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal i32 @__group_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 1
; CHECK-NEXT:    [[TMP3:%.*]] = udiv i32 [[TMP2]], 1
; CHECK-NEXT:    ret i32 [[TMP3]]

; CHECK-LABEL: define internal i32 @__group_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = urem i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = udiv i32 [[TMP1]], 1
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal ptr addrspace(5) @__global_offset_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS_GOFF]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret ptr addrspace(5) @__global_offset_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x__const

; CHECK-LABEL: define internal i32 @__global_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 20

; CHECK-LABEL: define internal i32 @__global_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__global_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__local_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 10

; CHECK-LABEL: define internal i32 @__local_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__local_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__local_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 10
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal i32 @__local_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 1
; CHECK-NEXT:    [[TMP3:%.*]] = urem i32 [[TMP2]], 1
; CHECK-NEXT:    ret i32 [[TMP3]]

; CHECK-LABEL: define internal i32 @__local_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = urem i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 1
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal i32 @__group_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = udiv i32 [[TMP1]], 10
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal i32 @__group_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = udiv i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = urem i32 [[TMP1]], 1
; CHECK-NEXT:    [[TMP3:%.*]] = udiv i32 [[TMP2]], 1
; CHECK-NEXT:    ret i32 [[TMP3]]

; CHECK-LABEL: define internal i32 @__group_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = urem i32 [[TMP0]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = udiv i32 [[TMP1]], 1
; CHECK-NEXT:    ret i32 [[TMP2]]

; CHECK-LABEL: define internal ptr addrspace(5) @__global_offset_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS_GOFF]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret ptr addrspace(5) @__global_offset_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_x__const

; CHECK-LABEL: define internal i32 @__global_size_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 10

; CHECK-LABEL: define internal i32 @__global_size_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__global_size_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__local_size_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 10

; CHECK-LABEL: define internal i32 @__local_size_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__local_size_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; This should be the last test

declare !sycl.kernel.fused !41 !sycl.kernel.nd-ranges !42 !sycl.kernel.nd-range !43 void @fused_kernel_homogeneous()

; CHECK-LABEL: define amdgpu_kernel void @fused_3(
; CHECK-NOT: remapper

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nounwind speculatable memory(none) }
attributes #2 = { nounwind }

; CHECK: attributes #[[ATTRS]] = { alwaysinline nocallback nofree nosync nounwind speculatable willreturn memory(none) }
; CHECK: attributes #[[ATTRS_GOFF]] = { alwaysinline nounwind speculatable memory(none) }

!6 = !{}
!11 = !{i32 64, i32 1, i32 1}
!12 = !{!"_arg_y"}
!13 = !{!"fused_0", !14}
!14 = !{!"KernelOne", !"KernelOne", !"KernelOne"}
!15 = !{!16, !17, !18}
!16 = !{i32 3, !19, !20, !21}
!17 = !{i32 2, !22, !20, !21}
!18 = !{i32 1, !23, !20, !21}
!19 = !{i64 2, i64 3, i64 7}
!20 = !{i64 2, i64 1, i64 1}
!21 = !{i64 0, i64 0, i64 0}
!22 = !{i64 2, i64 4, i64 1}
!23 = !{i64 48, i64 1, i64 1}
!24 = !{i32 3, !23, !20, !21}
!25 = !{!26, !27, !26}
!26 = !{i32 1, !28, !29, !21}
!27 = !{i32 1, !30, !29, !21}
!28 = !{i64 20, i64 1, i64 1}
!29 = !{i64 10, i64 1, i64 1}
!30 = !{i64 10, i64 1, i64 1}
!31 = !{!"fused_1", !14}
!32 = !{!"fused_2", !14}
!33 = !{!34, !35, !36}
!34 = !{i32 3, !37, !38, !21}
!35 = !{i32 3, !39, !38, !21}
!36 = !{i32 3, !40, !38, !21}
!37 = !{i64 60, i64 60, i64 60}
!38 = !{i64 2, i64 3, i64 20}
!39 = !{i64 2, i64 6, i64 40}
!40 = !{i64 6, i64 30, i64 60}
!41 = !{!"fused_3", !14}
!42 = !{!43, !43, !43}
!43 = !{i32 3, !44, !45, !46}
!44 = !{i64 100, i64 100, i64 100}
!45 = !{i64 10, i64 10, i64 10}
!46 = !{i64 0, i64 0, i64 0}
!47 = !{
  !"KernelOne",
  !{!"Accessor", !"StdLayout", !"StdLayout", !"StdLayout", !"Accessor",
    !"StdLayout", !"StdLayout", !"StdLayout", !"Accessor", !"StdLayout",
    !"StdLayout", !"StdLayout"},
  !{i8 1, i8 0, i8 0, i8 1, i8 1, i8 0, i8 0, i8 1, i8 1, i8 0, i8 0, i8 1},
  !{!"work_group_size_hint", i32 1, i32 1, i32 64}
}
!sycl.moduleinfo = !{!47}
