; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext\
; RUN: -passes=sycl-kernel-fusion -S %s | FileCheck %s

; This tests checks that kernel fusion fails when an unknown builtin
; is called inside a kernel.

; CHECK-NOT: fused_0

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

; Unsupported builtin
declare spir_func void @spirv_BuiltInWorkDim()

define spir_kernel void @KernelOne(i32 %x) #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_type_qual !0 !kernel_arg_base_type !0 !kernel_arg_name !0 !work_group_size_hint !1 {
entry:
  call spir_func void @spirv_BuiltInWorkDim()
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
