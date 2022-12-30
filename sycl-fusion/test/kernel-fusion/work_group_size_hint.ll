; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext\
; RUN: -passes=sycl-kernel-fusion\
; RUN: --sycl-info-path %S/work_group_size_hint.yaml -S %s\
; RUN: | FileCheck %s


target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"


; Function Attrs: alwaysinline nounwind
declare spir_func void @__itt_offload_wi_start_wrapper() #3

; Function Attrs: alwaysinline nounwind
declare spir_func void @__itt_offload_wi_finish_wrapper() #3


; Function Attrs: nounwind
define spir_kernel void @KernelOne(float addrspace(1)* align 4 %_arg_x) #2 !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_type_qual !9 !kernel_arg_base_type !8 !kernel_arg_name !10 !work_group_size_hint !11 {
entry:
  call spir_func void @__itt_offload_wi_start_wrapper() #3
  %0 = addrspacecast float addrspace(1)* %_arg_x to float addrspace(4)*
  store float 4.200000e+01, float addrspace(4)* %0, align 4
  call spir_func void @__itt_offload_wi_finish_wrapper() #3
  ret void
}

; Function Attrs: nounwind
define spir_kernel void @KernelTwo(float addrspace(1)* align 4 %_arg_y) #2 !kernel_arg_addr_space !6 !kernel_arg_access_qual !7 !kernel_arg_type !8 !kernel_arg_type_qual !9 !kernel_arg_base_type !8 !kernel_arg_name !12 {
entry:
  call spir_func void @__itt_offload_wi_start_wrapper() #3
  %0 = addrspacecast float addrspace(1)* %_arg_y to float addrspace(4)*
  store float 2.500000e+01, float addrspace(4)* %0, align 4
  call spir_func void @__itt_offload_wi_finish_wrapper() #3
  ret void
}

declare !sycl.kernel.fused !13 void @fused_kernel()

attributes #2 = { nounwind }
attributes #3 = { alwaysinline nounwind }

!6 = !{i32 1}
!7 = !{!"none"}
!8 = !{!"float*"}
!9 = !{!""}
!10 = !{!"_arg_x"}
!11 = !{i32 64, i32 1, i32 1}
!12 = !{!"_arg_y"}
!13 = !{!"fused_0", !14}
!14 = !{!"KernelOne", !"KernelTwo"}

; Test scenario: Fusion of two kernels where one of the kernels 
; specifies a "work_group_size_hint".
; This test focuses on the correct 'work_group_size_hint' metadata 
; being attached to the fused kernel.

; CHECK-LABEL: define spir_kernel void @fused_0
; CHECK-SAME: !work_group_size_hint ![[#REQD_SIZE:]]
;.
; CHECK: ![[#REQD_SIZE]] = !{i32 1, i32 1, i32 64}
;.
