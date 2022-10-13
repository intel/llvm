; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_float_controls2
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck %s --input-file %t.spt -check-prefix=SPV

; ModuleID = 'float_control.bc'
source_filename = "float_control.cpp"
target datalayout = "e-p:64:64-i64:64-n8:16:32"
target triple = "spir"

; Function Attrs: noinline norecurse nounwind readnone
define dso_local dllexport spir_kernel void @k_float_controls_0(i32 %ibuf, i32 %obuf) local_unnamed_addr {
entry:
  ret void
}

; Function Attrs: noinline norecurse nounwind readnone
define dso_local dllexport spir_kernel void @k_float_controls_1(i32 %ibuf, i32 %obuf) local_unnamed_addr {
entry:
  ret void
}

; Function Attrs: noinline norecurse nounwind readnone
define dso_local dllexport spir_kernel void @k_float_controls_2(i32 %ibuf, i32 %obuf) local_unnamed_addr {
entry:
  ret void
}

; Function Attrs: noinline norecurse nounwind readnone
define dso_local dllexport spir_kernel void @k_float_controls_3(i32 %ibuf, i32 %obuf) local_unnamed_addr {
entry:
  ret void
}


!llvm.module.flags = !{!12}
!llvm.ident = !{!13}
!spirv.EntryPoint = !{}
!spirv.ExecutionMode = !{!15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26}

; SPV-DAG: EntryPoint {{[0-9]+}} [[KERNEL0:[0-9]+]] "k_float_controls_0"
; SPV-DAG: EntryPoint {{[0-9]+}} [[KERNEL1:[0-9]+]] "k_float_controls_1"
; SPV-DAG: EntryPoint {{[0-9]+}} [[KERNEL2:[0-9]+]] "k_float_controls_2"
; SPV-DAG: EntryPoint {{[0-9]+}} [[KERNEL3:[0-9]+]] "k_float_controls_3"
!0 = !{void (i32, i32)* @k_float_controls_0, !"k_float_controls_0", !1, i32 0, !2, !3, !4, i32 0, i32 0}
!1 = !{i32 2, i32 2}
!2 = !{i32 32, i32 36}
!3 = !{i32 0, i32 0}
!4 = !{!"", !""}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 8.0.1"}
!14 = !{i32 1, i32 0}

; SPV-DAG: ExecutionMode [[KERNEL0]] 5620 64
!15 = !{void (i32, i32)* @k_float_controls_0, i32 5620, i32 64}
; SPV-DAG: ExecutionMode [[KERNEL0]] 5620 32
!16 = !{void (i32, i32)* @k_float_controls_0, i32 5620, i32 32}
; SPV-DAG: ExecutionMode [[KERNEL0]] 5620 16
!17 = !{void (i32, i32)* @k_float_controls_0, i32 5620, i32 16}

; SPV-DAG: ExecutionMode [[KERNEL1]] 5621 64
!18 = !{void (i32, i32)* @k_float_controls_1, i32 5621, i32 64}
; SPV-DAG: ExecutionMode [[KERNEL1]] 5621 32
!19 = !{void (i32, i32)* @k_float_controls_1, i32 5621, i32 32}
; SPV-DAG: ExecutionMode [[KERNEL1]] 5621 16
!20 = !{void (i32, i32)* @k_float_controls_1, i32 5621, i32 16}

; SPV-DAG: ExecutionMode [[KERNEL2]] 5622 64
!21 = !{void (i32, i32)* @k_float_controls_2, i32 5622, i32 64}
; SPV-DAG: ExecutionMode [[KERNEL2]] 5622 32
!22 = !{void (i32, i32)* @k_float_controls_2, i32 5622, i32 32}
; SPV-DAG: ExecutionMode [[KERNEL2]] 5622 16
!23 = !{void (i32, i32)* @k_float_controls_2, i32 5622, i32 16}

; SPV-DAG: ExecutionMode [[KERNEL3]] 5623 64
!24 = !{void (i32, i32)* @k_float_controls_3, i32 5623, i32 64}
; SPV-DAG: ExecutionMode [[KERNEL3]] 5623 32
!25 = !{void (i32, i32)* @k_float_controls_3, i32 5623, i32 32}
; SPV-DAG: ExecutionMode [[KERNEL3]] 5623 16
!26 = !{void (i32, i32)* @k_float_controls_3, i32 5623, i32 16}
