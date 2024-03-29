; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext\
; RUN: -passes=sycl-internalization -S %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK-LABEL: define {{[^@]+}}@fused_0
; CHECK-SAME: (ptr addrspace(1) align 4 %[[ACC:.*]])
define spir_kernel void @fused_0(ptr addrspace(1) align 4 %acc) !kernel_arg_addr_space !12 !kernel_arg_access_qual !13 !kernel_arg_type !14 !kernel_arg_type_qual !15 !kernel_arg_base_type !14 !kernel_arg_name !16 !sycl.kernel.promote !17 !sycl.kernel.promote.localsize !18 {
; Scenario: Test private internalization is not performed when the
; input pointer is stored in another pointer.

; CHECK-NEXT:  %[[ALLOCA:.*]] = alloca ptr addrspace(1), align 8
; CHECK-NEXT:  store ptr addrspace(1) %[[ACC]], ptr %[[ALLOCA]], align 8
; CHECK-NEXT:  %[[ACC_PTR:.*]] = load ptr addrspace(1), ptr %[[ALLOCA]], align 8
; CHECK-NEXT:  store float 7.000000e+00, ptr addrspace(1) %[[ACC]], align 4
; CHECK-NEXT:  %[[RES:.*]] = load float, ptr addrspace(1) %[[ACC]], align 4
; CHECK-NEXT:  ret void

  %alloca = alloca ptr addrspace(1)
  store ptr addrspace(1) %acc, ptr %alloca
  %acc_ptr = load ptr addrspace(1), ptr %alloca
  store float 7.0, ptr addrspace(1) %acc
  %res = load float, ptr addrspace(1) %acc
  ret void
}

!12 = !{i32 1}
!13 = !{!"none"}
!14 = !{!"ptr"}
!15 = !{!""}
!16 = !{!"acc"}
!17 = !{!"private"}
!18 = !{i64 1}
