; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext \
; RUN:   -passes=sycl-kernel-fusion --sycl-info-path %S/check-remapping.yaml \
; RUN:   -S %s | FileCheck %s

; This tests checks that PTX intrinsics are correctly remapped when fusing
; kernels with different ND-ranges.

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8"
target triple = "amdgcn-amd-amdhsa"

declare noundef i32 @llvm.amdgcn.workitem.id.x() #0
declare noundef i32 @llvm.amdgcn.workitem.id.y() #0
declare noundef i32 @llvm.amdgcn.workitem.id.z() #0
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #0
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #0
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #0
declare ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #1
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
  %6 = call ptr @llvm.amdgcn.implicit.offset() #1
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
  %10 = zext i16 %l4 to i64
  ; Workgroup size, y
  %arrayidx5 = getelementptr inbounds i16, ptr addrspace(4) %DPtr, i64 3
  %l5 = load i16, ptr addrspace(4) %arrayidx5, align 4
  %11 = zext i16 %l5 to i64
  ; Workgroup size, z
  %arrayidx6 = getelementptr inbounds i16, ptr addrspace(4) %DPtr, i64 4
  %l6 = load i16, ptr addrspace(4) %arrayidx6, align 4
  %12 = zext i16 %l6 to i64
  ret void
}

declare !sycl.kernel.fused !13 !sycl.kernel.nd-ranges !15 !sycl.kernel.nd-range !24 void @fused_kernel()

; CHECK: @__global_offset_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1__const = internal constant [3 x i32] zeroinitializer
; CHECK: @__global_offset_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1__const = internal constant [3 x i32] zeroinitializer
; CHECK: @__global_offset_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1__const = internal constant [3 x i32] zeroinitializer
; CHECK: @__global_offset_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1__const = internal constant [3 x i32] zeroinitializer
; CHECK: @__global_offset_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1__const = internal constant [3 x i32] zeroinitializer

; CHECK-LABEL: define void @fused_0(
; CHECK-SAME: i32 [[KERNELONE_X:%.*]], i32 [[KERNELONE_X1:%.*]], i32 [[KERNELONE_X2:%.*]]) !work_group_size_hint !3 !kernel_arg_buffer_location !4 !kernel_arg_runtime_aligned !4 !kernel_arg_exclusive_ptr !4 {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS:.*]]
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ult i32 [[TMP0]], 42
; CHECK-NEXT:    br i1 [[TMP1]], label [[TMP2:%.*]], label [[TMP16:%.*]]
; CHECK:       2:
; CHECK-NEXT:    [[TMP3:%.*]] = call i32 @__local_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x()  #[[ATTRS]]
; CHECK-NEXT:    [[TMP4:%.*]] = call i32 @__local_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP5:%.*]] = call i32 @__local_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP6:%.*]] = call i32 @__local_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP7:%.*]] = call i32 @__local_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP8:%.*]] = call i32 @__local_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP9:%.*]] = call i32 @__group_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP10:%.*]] = call i32 @__group_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP11:%.*]] = call i32 @__group_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP12:%.*]] = call i32 @__num_work_groups_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP13:%.*]] = call i32 @__num_work_groups_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP14:%.*]] = call i32 @__num_work_groups_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP15:%.*]] = call ptr @__global_offset_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1() #[[ATTRS_GOFF:.*]]
; CHECK-NEXT:    br label [[TMP16]]
; CHECK:       16:
; CHECK-NEXT:    call void @llvm.nvvm.barrier0()
; CHECK-NEXT:    [[TMP17:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP18:%.*]] = icmp ult i32 [[TMP17]], 8
; CHECK-NEXT:    br i1 [[TMP18]], label [[TMP19:%.*]], label [[TMP33:%.*]]
; CHECK:       19:
; CHECK-NEXT:    [[TMP20:%.*]] = call i32 @__local_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP21:%.*]] = call i32 @__local_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP22:%.*]] = call i32 @__local_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP23:%.*]] = call i32 @__local_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP24:%.*]] = call i32 @__local_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP25:%.*]] = call i32 @__local_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP26:%.*]] = call i32 @__group_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP27:%.*]] = call i32 @__group_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP28:%.*]] = call i32 @__group_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP29:%.*]] = call i32 @__num_work_groups_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP30:%.*]] = call i32 @__num_work_groups_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP31:%.*]] = call i32 @__num_work_groups_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP32:%.*]] = call ptr @__global_offset_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1() #[[ATTRS_GOFF]]
; CHECK-NEXT:    br label [[TMP33]]
; CHECK:       33:
; CHECK-NEXT:    call void @llvm.nvvm.barrier0()
; CHECK-NEXT:    [[TMP34:%.*]] = call i32 @__local_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP35:%.*]] = call i32 @__local_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP36:%.*]] = call i32 @__local_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP37:%.*]] = call i32 @__local_size_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP38:%.*]] = call i32 @__local_size_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP39:%.*]] = call i32 @__local_size_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP40:%.*]] = call i32 @__group_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP41:%.*]] = call i32 @__group_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP42:%.*]] = call i32 @__group_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP43:%.*]] = call i32 @__num_work_groups_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP44:%.*]] = call i32 @__num_work_groups_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP45:%.*]] = call i32 @__num_work_groups_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP46:%.*]] = call ptr @__global_offset_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1() #[[ATTRS_GOFF]]
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
; CHECK-NEXT:    [[TMP1:%.*]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
; CHECK-NEXT:    [[TMP2:%.*]] = call i32 @llvm.amdgcn.workitem.id.z()
; CHECK-NEXT:    [[TMP3:%.*]] = mul i32 [[TMP0]], [[TMP1]]
; CHECK-NEXT:    [[TMP4:%.*]] = add i32 [[TMP3]], [[TMP2]]
; CHECK-NEXT:    ret i32 [[TMP4]]

; CHECK-LABEL: define internal i32 @__global_id_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @llvm.amdgcn.workgroup.id.y()
; CHECK-NEXT:    [[TMP1:%.*]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
; CHECK-NEXT:    [[TMP2:%.*]] = call i32 @llvm.amdgcn.workitem.id.y()
; CHECK-NEXT:    [[TMP3:%.*]] = mul i32 [[TMP0]], [[TMP1]]
; CHECK-NEXT:    [[TMP4:%.*]] = add i32 [[TMP3]], [[TMP2]]
; CHECK-NEXT:    ret i32 [[TMP4]]

; CHECK-LABEL: define internal i32 @__global_id_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @llvm.amdgcn.workgroup.id.x()
; CHECK-NEXT:    [[TMP1:%.*]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
; CHECK-NEXT:    [[TMP2:%.*]] = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK-NEXT:    [[TMP3:%.*]] = mul i32 [[TMP0]], [[TMP1]]
; CHECK-NEXT:    [[TMP4:%.*]] = add i32 [[TMP3]], [[TMP2]]
; CHECK-NEXT:    ret i32 [[TMP4]]

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

; CHECK-LABEL: define internal i32 @__num_work_groups_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 7

; CHECK-LABEL: define internal i32 @__num_work_groups_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 3

; CHECK-LABEL: define internal i32 @__num_work_groups_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal ptr @__global_offset_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME: ) #[[ATTRS_GOFF]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret ptr @__global_offset_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1__const

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

; CHECK-LABEL: define internal i32 @__num_work_groups_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 4

; CHECK-LABEL: define internal i32 @__num_work_groups_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__num_work_groups_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal ptr @__global_offset_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME: ) #[[ATTRS_GOFF]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret ptr @__global_offset_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1__const

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

; CHECK-LABEL: define internal i32 @__num_work_groups_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 24

; CHECK-LABEL: define internal i32 @__num_work_groups_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__num_work_groups_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal ptr @__global_offset_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME: ) #[[ATTRS_GOFF]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret ptr @__global_offset_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1__const

declare !sycl.kernel.fused !31 !sycl.kernel.nd-ranges !25 !sycl.kernel.nd-range !24 void @fused_kernel_1D()

; CHECK-LABEL: define void @fused_1(
; CHECK-SAME: i32 [[KERNELONE_X:%.*]], i32 [[KERNELONE_X1:%.*]], i32 [[KERNELONE_X2:%.*]]) !work_group_size_hint !3 !kernel_arg_buffer_location !4 !kernel_arg_runtime_aligned !4 !kernel_arg_exclusive_ptr !4 {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ult i32 [[TMP0]], 20
; CHECK-NEXT:    br i1 [[TMP1]], label [[TMP2:%.*]], label [[TMP16:%.*]]
; CHECK:       2:
; CHECK-NEXT:    [[TMP3:%.*]] = call i32 @__local_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP4:%.*]] = call i32 @__local_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP5:%.*]] = call i32 @__local_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP6:%.*]] = call i32 @__local_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP7:%.*]] = call i32 @__local_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP8:%.*]] = call i32 @__local_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP9:%.*]] = call i32 @__group_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP10:%.*]] = call i32 @__group_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP11:%.*]] = call i32 @__group_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP12:%.*]] = call i32 @__num_work_groups_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP13:%.*]] = call i32 @__num_work_groups_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP14:%.*]] = call i32 @__num_work_groups_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP15:%.*]] = call ptr @__global_offset_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1() #[[ATTRS_GOFF]]
; CHECK-NEXT:    br label [[TMP16]]
; CHECK:       16:
; CHECK-NEXT:    call void @llvm.nvvm.barrier0()
; CHECK-NEXT:    [[TMP17:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP18:%.*]] = icmp ult i32 [[TMP17]], 10
; CHECK-NEXT:    br i1 [[TMP18]], label [[TMP19:%.*]], label [[TMP33:%.*]]
; CHECK:       19:
; CHECK-NEXT:    [[TMP20:%.*]] = call i32 @__local_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP21:%.*]] = call i32 @__local_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP22:%.*]] = call i32 @__local_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP23:%.*]] = call i32 @__local_size_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP24:%.*]] = call i32 @__local_size_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP25:%.*]] = call i32 @__local_size_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP26:%.*]] = call i32 @__group_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP27:%.*]] = call i32 @__group_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP28:%.*]] = call i32 @__group_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP29:%.*]] = call i32 @__num_work_groups_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP30:%.*]] = call i32 @__num_work_groups_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP31:%.*]] = call i32 @__num_work_groups_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP32:%.*]] = call ptr @__global_offset_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1() #[[ATTRS_GOFF]]
; CHECK-NEXT:    br label [[TMP33]]
; CHECK:       33:
; CHECK-NEXT:    call void @llvm.nvvm.barrier0()
; CHECK-NEXT:    [[TMP34:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    [[TMP35:%.*]] = icmp ult i32 [[TMP34]], 20
; CHECK-NEXT:    br i1 [[TMP35]], label [[TMP36:%.*]], label [[TMP50:%.*]]
; CHECK:       36:
; CHECK-NEXT:    [[TMP37:%.*]] = call i32 @__local_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP38:%.*]] = call i32 @__local_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP39:%.*]] = call i32 @__local_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP40:%.*]] = call i32 @__local_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP41:%.*]] = call i32 @__local_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP42:%.*]] = call i32 @__local_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP43:%.*]] = call i32 @__group_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP44:%.*]] = call i32 @__group_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP45:%.*]] = call i32 @__group_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP46:%.*]] = call i32 @__num_work_groups_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP47:%.*]] = call i32 @__num_work_groups_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP48:%.*]] = call i32 @__num_work_groups_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP49:%.*]] = call ptr @__global_offset_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1() #[[ATTRS_GOFF]]
; CHECK-NEXT:    br label [[TMP50]]
; CHECK:       50:
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

; CHECK-LABEL: define internal i32 @__num_work_groups_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 2

; CHECK-LABEL: define internal i32 @__num_work_groups_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__num_work_groups_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal ptr @__global_offset_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1(
; CHECK-SAME: ) #[[ATTRS_GOFF]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret ptr @__global_offset_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1__const

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

; CHECK-LABEL: define internal i32 @__num_work_groups_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_x(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__num_work_groups_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_y(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal i32 @__num_work_groups_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1_z(
; CHECK-SAME: ) #[[ATTRS]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i32 1

; CHECK-LABEL: define internal ptr @__global_offset_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1(
; CHECK-SAME: ) #[[ATTRS_GOFF]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret ptr @__global_offset_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1__const

; This should be the last test

declare !sycl.kernel.fused !41 !sycl.kernel.nd-ranges !42 !sycl.kernel.nd-range !43 void @fused_kernel_homogeneous()

; CHECK-LABEL: define void @fused_3(
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
