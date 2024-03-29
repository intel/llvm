; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext \
; RUN: -passes=sycl-internalization -S %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: noinline
define void @fun(ptr addrspace(1) %arg) #0 {
  ret void
}

%struct = type { i32, i32, i32 }

; CHECK-LABEL: define {{[^@]+}}@fused_0
; CHECK-SAME: (ptr addrspace(1) align 4 %[[ACC:.*]])
define spir_kernel void @fused_0(ptr addrspace(1) align 4 %acc) !kernel_arg_addr_space !12 !kernel_arg_access_qual !13 !kernel_arg_type !14 !kernel_arg_type_qual !15 !kernel_arg_base_type !14 !kernel_arg_name !16 !sycl.kernel.promote !17 !sycl.kernel.promote.localsize !18 {
; Scenario: Test private internalization is not performed when pointers into
; aggregate object are passed to function calls.

; CHECK-NOT:  alloca [1 x %struct]
  %gep1 = getelementptr %struct, ptr addrspace(1) %acc, i64 17
  %gep2 = getelementptr %struct, ptr addrspace(1) %gep1, i64 0, i32 2
  call void @fun(ptr addrspace(1) %gep2)
  store i32 42, ptr addrspace(1) %gep2
  ret void
}

attributes #0 = { noinline }

!12 = !{i32 1}
!13 = !{!"none"}
!14 = !{!"ptr"}
!15 = !{!""}
!16 = !{!"acc"}
!17 = !{!"private"}
!18 = !{i64 1}
