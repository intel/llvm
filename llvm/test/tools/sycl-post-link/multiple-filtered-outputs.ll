; This checks that sycl-post-link can accept multiple -o options, 
; with some of the -o options being composed of a (target, filename) pair,
; and that the output tables from inputs with target info have the modules
; that are not compatible with that target filtered out.

; RUN: sycl-post-link -properties %s -symbols -split=auto \
; RUN: -o %t.table \
; RUN: -o intel_gpu_pvc,%t-pvc.table \
; RUN: -o intel_gpu_tgllp,%t-tgllp.table \
; RUN: -o intel_gpu_cfl,%t-cfl.table \
; RUN: -o unrecognized_target,%t-unrecognized.table

; RUN: FileCheck %s -input-file=%t_0.sym -check-prefix=CHECK-DOUBLE
; RUN: FileCheck %s -input-file=%t_1.sym -check-prefix=CHECK-SG8
; RUN: FileCheck %s -input-file=%t_2.sym -check-prefix=CHECK-SG64
; RUN: FileCheck %s -input-file=%t_3.sym -check-prefix=CHECK-SG32
; RUN: FileCheck %s -input-file=%t_4.sym -check-prefix=CHECK-FLOAT
; RUN: FileCheck %s -input-file=%t_5.sym -check-prefix=CHECK-SG16

; RUN: FileCheck %s -input-file=%t.table -check-prefix=CHECK-ALL
; RUN: FileCheck %s -input-file=%t-unrecognized.table -check-prefix=CHECK-ALL
; RUN: FileCheck %s -input-file=%t-pvc.table -check-prefix=CHECK-PVC
; RUN: FileCheck %s -input-file=%t-tgllp.table -check-prefix=CHECK-TGLLP
; RUN: FileCheck %s -input-file=%t-cfl.table -check-prefix=CHECK-CFL

; CHECK-DOUBLE: double_kernel
; CHECK-FLOAT: float_kernel
; CHECK-SG8: reqd_sub_group_size_kernel_8
; CHECK-SG16: reqd_sub_group_size_kernel_16
; CHECK-SG32: reqd_sub_group_size_kernel_32
; CHECK-SG64: reqd_sub_group_size_kernel_64

; An output without a target will have no filtering performed on the output table.
; Additionally, an unrecognized target will perform the same.
; CHECK-ALL:      _0.sym
; CHECK-ALL-NEXT: _1.sym
; CHECK-ALL-NEXT: _2.sym
; CHECK-ALL-NEXT: _3.sym
; CHECK-ALL-NEXT: _4.sym
; CHECK-ALL-NEXT: _5.sym
; CHECK-ALL-EMPTY:

; PVC does not support sg8 (=1) or sg64 (=2) 
; CHECK-PVC:      _0.sym
; CHECK-PVC-NEXT: _3.sym
; CHECK-PVC-NEXT: _4.sym
; CHECK-PVC-NEXT: _5.sym
; CHECK-PVC-EMPTY:

; TGLLP does not support fp64 (=0) or sg64 (=2)
; CHECK-TGLLP:      _1.sym
; CHECK-TGLLP-NEXT: _3.sym
; CHECK-TGLLP-NEXT: _4.sym
; CHECK-TGLLP-NEXT: _5.sym
; CHECK-TGLLP-EMPTY:

; CFL does not support sg64 (=2)
; CHECK-CFL:      _0.sym
; CHECK-CFL-NEXT: _1.sym
; CHECK-CFL-NEXT: _3.sym
; CHECK-CFL-NEXT: _4.sym
; CHECK-CFL-NEXT: _5.sym
; CHECK-CFL-EMPTY:

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_kernel void @double_kernel(ptr addrspace(1) noundef align 8 %_arg_out) #0 !sycl_used_aspects !67 {
entry:
  %0 = load double, ptr addrspace(1) %_arg_out, align 8
  %mul.i = fmul double %0, 2.000000e-01
  store double %mul.i, ptr addrspace(1) %_arg_out, align 8
  ret void
}

define spir_kernel void @float_kernel(ptr addrspace(1) noundef align 4 %_arg_out) #0 {
entry:
  %0 = load float, ptr addrspace(1) %_arg_out, align 4
  %mul.i = fmul float %0, 0x3FC99999A0000000
  store float %mul.i, ptr addrspace(1) %_arg_out, align 4
  ret void
}

define spir_kernel void @reqd_sub_group_size_kernel_8() #0 !intel_reqd_sub_group_size !78 {
entry:
  ret void
}

define spir_kernel void @reqd_sub_group_size_kernel_16() #0 !intel_reqd_sub_group_size !79 {
entry:
  ret void
}

define spir_kernel void @reqd_sub_group_size_kernel_32() #0 !intel_reqd_sub_group_size !80 {
entry:
  ret void
}

define spir_kernel void @reqd_sub_group_size_kernel_64() #0 !intel_reqd_sub_group_size !81 {
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
!9 = !{!"fp64", i32 6}
!64 = !{!"clang version 19.0.0git (/ws/llvm/clang a7f3a637bdd6299831f903bbed9e8d069fea5c86)"}
!67 = !{!9}
!78 = !{i32 8}
!79 = !{i32 16}
!80 = !{i32 32}
!81 = !{i32 64}
