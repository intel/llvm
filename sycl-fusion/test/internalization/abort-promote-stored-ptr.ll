; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext\
; RUN: -passes=sycl-internalization --sycl-info-path %S/abort-kernel-info.yaml -S %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK-LABEL: define {{[^@]+}}@fused_0
; CHECK-SAME: (float addrspace(1)* align 4 %[[ACC:.*]])
define spir_kernel void @fused_0(float addrspace(1)* align 4 %acc) !kernel_arg_addr_space !12 !kernel_arg_access_qual !13 !kernel_arg_type !14 !kernel_arg_type_qual !15 !kernel_arg_base_type !14 !kernel_arg_name !16 !sycl.kernel.promote !17 !sycl.kernel.promote.localsize !18 {
; Scenario: Test private internalization is not performed when the
; input pointer is stored in another pointer.

; CHECK-NEXT:  %[[ALLOCA:.*]] = alloca float addrspace(1)*, align 8
; CHECK-NEXT:  store float addrspace(1)* %[[ACC]], float addrspace(1)** %[[ALLOCA]], align 8
; CHECK-NEXT:  %[[ACC_PTR:.*]] = load float addrspace(1)*, float addrspace(1)** %[[ALLOCA]], align 8
; CHECK-NEXT:  store float 7.000000e+00, float addrspace(1)* %[[ACC]], align 4
; CHECK-NEXT:  %[[RES:.*]] = load float, float addrspace(1)* %[[ACC]], align 4
; CHECK-NEXT:  ret void

  %alloca = alloca float addrspace(1)*
  store float addrspace(1)* %acc, float addrspace(1)** %alloca
  %acc_ptr = load float addrspace(1)*, float addrspace(1)** %alloca
  store float 7.0, float addrspace(1)* %acc
  %res = load float, float addrspace(1)* %acc
  ret void
}

!12 = !{i32 1}
!13 = !{!"none"}
!14 = !{!"float*"}
!15 = !{!""}
!16 = !{!"acc"}
!17 = !{!"private"}
!18 = !{i64 1}
