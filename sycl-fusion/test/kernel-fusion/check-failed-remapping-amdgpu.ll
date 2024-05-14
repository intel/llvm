; REQUIRES: hip_amd
; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext \
; RUN:   -passes=sycl-kernel-fusion -S %s | FileCheck %s

; This tests checks that kernel fusion fails when a not-remappable AMDGPU
; intrinsic is called inside a kernel.

; CHECK-NOT: fused_0

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8"
target triple = "amdgcn-amd-amdhsa"

; Unsupported intrinsic
declare noundef i32 @llvm.amdgcn.mbcnt.hi(i32 %arg0, i32 %arg1)

define spir_kernel void @KernelOne(i32 %x) #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_type_qual !0 !kernel_arg_base_type !0 !kernel_arg_name !0 !work_group_size_hint !1 {
entry:
  %0 = call i32 @llvm.amdgcn.mbcnt.hi(i32 %x, i32 %x)
  ret void
}

declare !sycl.kernel.fused !2 !sycl.kernel.nd-ranges !4 !sycl.kernel.nd-range !13 void @fused_kernel()

attributes #0 = { nounwind }

!0 = !{}
!1 = !{i32 64, i32 1, i32 1}
!2 = !{!"fused_0", !3}
!3 = !{!"KernelOne", !"KernelOne", !"KernelOne"}
!4 = !{!5, !9, !11}
!5 = !{i32 3, !6, !7, !8}
!6 = !{i64 2, i64 3, i64 7}
!7 = !{i64 2, i64 1, i64 1}
!8 = !{i64 0, i64 0, i64 0}
!9 = !{i32 2, !10, !7, !8}
!10 = !{i64 2, i64 4, i64 1}
!11 = !{i32 1, !12, !7, !8}
!12 = !{i64 48, i64 1, i64 1}
!13 = !{i32 3, !12, !7, !8}
!14 = !{
  !"KernelOne",
  !{!"Accessor", !"StdLayout", !"StdLayout", !"StdLayout", !"Accessor",
    !"StdLayout", !"StdLayout", !"StdLayout", !"Accessor", !"StdLayout",
    !"StdLayout", !"StdLayout"},
  !{i8 1, i8 0, i8 0, i8 1, i8 1, i8 0, i8 0, i8 1, i8 1, i8 0, i8 0, i8 1},
  !{!"work_group_size_hint", i32 1, i32 1, i32 64}
}
!sycl.moduleinfo = !{!14}
