; With ESIMD, the reqd_sub_group_size of a kernel will be 1. Normally,
; no device can handled compiling for this reqd_sub_group_size, but 
; for ESIMD, this is an exception. This test makes sure that 
; ESIMD kernels are not filtered out when using filtering 
; (e.g. -o intel_gpu_dg1,%t-dg1.table) and also ensures that 
; non ESIMD kernels with reqd_sub_group_size=1 are still filtered out.

; RUN: sycl-post-link %s -symbols -split=auto \
; RUN: -o intel_gpu_dg1,%t-dg1.table

; RUN: FileCheck %s -input-file=%t-dg1.table -check-prefix=CHECK-TABLE
; RUN: FileCheck %s -input-file=%t-dg1_esimd_0.sym -check-prefix=CHECK-SYM -implicit-check-not=reqd_sub_group_size_kernel_1

; CHECK-TABLE: _esimd_0.sym
; CHECK-SYM: esimd_kernel

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_kernel void @esimd_kernel(ptr addrspace(1) noundef align 8 %_arg_out) #0 !sycl_explicit_simd !69 !intel_reqd_sub_group_size !68 !sycl_used_aspects !67 {
entry:
  ret void
}

define spir_kernel void @reqd_sub_group_size_kernel_1(ptr addrspace(1) noundef align 8 %_arg_out) #0 !intel_reqd_sub_group_size !68 {
entry:
  ret void
}

attributes #0 = { mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="double.cpp" "sycl-optlevel"="3" "uniform-work-group-size"="true" }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!llvm.ident = !{!64}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!9 = !{!"ext_intel_esimd", i32 53}
!64 = !{!"clang version 19.0.0git (/ws/llvm/clang a7f3a637bdd6299831f903bbed9e8d069fea5c86)"}
!67 = !{!9}
!68 = !{i32 1}
!69 = !{}
!78 = !{i32 8}
!79 = !{i32 16}
!80 = !{i32 32}
!81 = !{i32 64}
