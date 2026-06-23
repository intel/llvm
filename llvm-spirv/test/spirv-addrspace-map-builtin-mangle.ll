; Test --spirv-addrspace-map address space remapping in SPIR-V builtin
; function names for opaque (image) types.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r --spirv-target-env=SPV-IR %t.spv -o - \
; RUN:   | llvm-dis | FileCheck %s --check-prefix=CHECK-DEFAULT
; RUN: llvm-spirv -r --spirv-target-env=SPV-IR --spirv-addrspace-map=1:3 %t.spv -o - \
; RUN:   | llvm-dis | FileCheck %s --check-prefix=CHECK-MAPPED

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

define spir_kernel void @write_kernel(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %img, <2 x i32> %coord, <4 x float> %val) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
; Without an address space map, image types use global (AS1): U3AS1 in the name.
; CHECK-DEFAULT: __spirv_ImageWritePU3AS1
; With mapping 1->3 (global->local), U3AS3 must appear instead.
; CHECK-MAPPED: __spirv_ImageWritePU3AS3
; CHECK-MAPPED-NOT: __spirv_ImageWritePU3AS1
  call spir_func void @_Z12write_imagef14ocl_image2d_woDv2_iDv4_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %img, <2 x i32> %coord, <4 x float> %val)
  ret void
}

declare spir_func void @_Z12write_imagef14ocl_image2d_woDv2_iDv4_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1), <2 x i32>, <4 x float>)

attributes #0 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!6}
!opencl.used.extensions = !{!7}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!7}

!1 = !{i32 1, i32 0, i32 0}
!2 = !{!"write_only", !"none", !"none"}
!3 = !{!"image2d_t", !"int2", !"float4"}
!4 = !{!"image2d_t", !"int2", !"float4"}
!5 = !{!"", !"", !""}
!6 = !{i32 2, i32 0}
!7 = !{}
!8 = !{!"cl_images"}
