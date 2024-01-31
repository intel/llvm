; REQUIRES: cuda
; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext \
; RUN:   -passes=sycl-kernel-fusion -S %s | FileCheck %s

; This tests checks that PTX intrinsics are correctly remapped when fusing
; kernels with different ND-ranges.

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #0
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.z() #0
declare noundef i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
declare noundef i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #0
declare noundef i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #0
declare ptr @llvm.nvvm.implicit.offset() #1

define void @foo(i32 %x) {
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #0
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z() #0
  %9 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
  %10 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #0
  %11 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #0
  %12 = call ptr @llvm.nvvm.implicit.offset() #1
  ret void
}

define i32 @bar(i32 %x) {
entry:
  %cmp = icmp ule i32 %x, 1
  br i1 %cmp, label %return, label %if.end

if.end:
  %sub = sub i32 %x, 1
  %call = call i32 @bar(i32 %sub)
  %mul = mul i32 %x, %call
  br label %return

return:
  %res = phi i32 [%x, %entry], [%sub, %if.end]
  ret i32 %res
}

define void @baz(i32 %x) {
entry:
  ret void
}

; Function Attrs: nounwind
define spir_kernel void @KernelOne(i32 %x) #2 !kernel_arg_addr_space !6 !kernel_arg_access_qual !6 !kernel_arg_type !6 !kernel_arg_type_qual !6 !kernel_arg_base_type !6 !kernel_arg_name !6 !work_group_size_hint !11 {
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #0
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z() #0
  %9 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
  %10 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #0
  %11 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #0
  %12 = call ptr @llvm.nvvm.implicit.offset() #1
  call void @foo(i32 %x)
  %y = call i32 @bar(i32 %x)
  call void @baz(i32 %y)
  ret void
}

declare !sycl.kernel.fused !13 !sycl.kernel.nd-ranges !15 !sycl.kernel.nd-range !24 void @fused_kernel()

; CHECK-LABEL: define void @fused_0(
; CHECK-SAME: i32 [[KERNELONE_X:%.*]], i32 [[KERNELONE_X1:%.*]], i32 [[KERNELONE_X2:%.*]]) !work_group_size_hint !3 !kernel_arg_buffer_location !4 !kernel_arg_runtime_aligned !4 !kernel_arg_exclusive_ptr !4 {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS:.*]]
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ult i32 [[TMP0]], 42
; CHECK-NEXT:    br i1 [[TMP1]], label [[TMP2:%.*]], label [[TMP16:%.*]]
; CHECK:       2:
; CHECK-NEXT:    [[TMP3:%.*]] = call i32 @__local_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
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
; CHECK-NEXT:    call void @foo2(i32 [[KERNELONE_X]]) #[[ATTR5:[0-9]+]]
; CHECK-NEXT:    [[Y_I:%.*]] = call i32 @bar3(i32 [[KERNELONE_X]]) #[[ATTR5]]
; CHECK-NEXT:    call void @baz(i32 [[Y_I]]) #[[ATTR5]]
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
; CHECK-NEXT:    call void @foo6(i32 [[KERNELONE_X1]]) #[[ATTR5]]
; CHECK-NEXT:    [[Y_I3:%.*]] = call i32 @bar7(i32 [[KERNELONE_X1]]) #[[ATTR5]]
; CHECK-NEXT:    call void @baz(i32 [[Y_I3]]) #[[ATTR5]]
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
; CHECK-NEXT:    call void @foo10(i32 [[KERNELONE_X2]]) #[[ATTR5]]
; CHECK-NEXT:    [[Y_I4:%.*]] = call i32 @bar11(i32 [[KERNELONE_X2]]) #[[ATTR5]]
; CHECK-NEXT:    call void @baz(i32 [[Y_I4]]) #[[ATTR5]]
; CHECK-NEXT:    ret void

; CHECK-LABEL: define void @foo2(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__local_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = call i32 @__local_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP2:%.*]] = call i32 @__local_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP3:%.*]] = call i32 @__local_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP4:%.*]] = call i32 @__local_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP5:%.*]] = call i32 @__local_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP6:%.*]] = call i32 @__group_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP7:%.*]] = call i32 @__group_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP8:%.*]] = call i32 @__group_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP9:%.*]] = call i32 @__num_work_groups_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP10:%.*]] = call i32 @__num_work_groups_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP11:%.*]] = call i32 @__num_work_groups_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP12:%.*]] = call ptr @__global_offset_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1() #[[ATTRS_GOFF]]
; CHECK-NEXT:    ret void

; CHECK-LABEL: define i32 @bar3(
; CHECK:         call i32 @bar3

; CHECK-LABEL: define void @foo6(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @__local_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP1:%.*]] = call i32 @__local_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP2:%.*]] = call i32 @__local_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP3:%.*]] = call i32 @__local_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP4:%.*]] = call i32 @__local_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP5:%.*]] = call i32 @__local_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP6:%.*]] = call i32 @__group_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP7:%.*]] = call i32 @__group_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP8:%.*]] = call i32 @__group_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP9:%.*]] = call i32 @__num_work_groups_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_x() #[[ATTRS]]
; CHECK-NEXT:    [[TMP10:%.*]] = call i32 @__num_work_groups_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_y() #[[ATTRS]]
; CHECK-NEXT:    [[TMP11:%.*]] = call i32 @__num_work_groups_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1_z() #[[ATTRS]]
; CHECK-NEXT:    [[TMP12:%.*]] = call ptr @__global_offset_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1() #[[ATTRS_GOFF]]
; CHECK-NEXT:    ret void

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
; CHECK-NEXT:    call void @foo14(i32 [[KERNELONE_X]]) #[[ATTR5]]
; CHECK-NEXT:    [[Y_I:%.*]] = call i32 @bar15(i32 [[KERNELONE_X]]) #[[ATTR5]]
; CHECK-NEXT:    call void @baz(i32 [[Y_I]]) #[[ATTR5]]
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
; CHECK-NEXT:    call void @foo18(i32 [[KERNELONE_X1]]) #[[ATTR5]]
; CHECK-NEXT:    [[Y_I3:%.*]] = call i32 @bar19(i32 [[KERNELONE_X1]]) #[[ATTR5]]
; CHECK-NEXT:    call void @baz(i32 [[Y_I3]]) #[[ATTR5]]
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
; CHECK-NEXT:    call void @foo14(i32 [[KERNELONE_X2]]) #[[ATTR5]]
; CHECK-NEXT:    [[Y_I4:%.*]] = call i32 @bar15(i32 [[KERNELONE_X2]]) #[[ATTR5]]
; CHECK-NEXT:    call void @baz(i32 [[Y_I4]]) #[[ATTR5]]
; CHECK-NEXT:    br label [[TMP50]]
; CHECK:       50:
; CHECK-NEXT:    ret void

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
!47 = !{
  !"KernelOne",
  !{!"Accessor", !"StdLayout", !"StdLayout", !"StdLayout", !"Accessor",
    !"StdLayout", !"StdLayout", !"StdLayout", !"Accessor", !"StdLayout",
    !"StdLayout", !"StdLayout"},
  !{i8 1, i8 0, i8 0, i8 1, i8 1, i8 0, i8 0, i8 1, i8 1, i8 0, i8 0, i8 1},
  !{!"work_group_size_hint", i32 1, i32 1, i32 64}
}
!sycl.moduleinfo = !{!47}
