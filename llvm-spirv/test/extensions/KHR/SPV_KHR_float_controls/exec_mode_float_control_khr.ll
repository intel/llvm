; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-max-version=1.1 --spirv-ext=+SPV_KHR_float_controls
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck %s --input-file %t.spt -check-prefixes=SPV,SPVEXT
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-max-version=1.4
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck %s --input-file %t.spt -check-prefixes=SPV,SPV14
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-max-version=1.1
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck %s --input-file %t.spt -check-prefix=SPV-NEGATIVE

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

; Function Attrs: noinline norecurse nounwind readnone
define dso_local dllexport spir_kernel void @k_float_controls_4(i32 %ibuf, i32 %obuf) local_unnamed_addr {
entry:
  ret void
}


!llvm.module.flags = !{!12}
!llvm.ident = !{!13}
!spirv.EntryPoint = !{}
!spirv.ExecutionMode = !{!15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29}

; SPVEXT-DAG: Extension "SPV_KHR_float_controls"
; SPV14-NOT: Extension "SPV_KHR_float_controls"
; SPV-NEGATIVE-NOT: Extension "SPV_KHR_float_controls"

; SPV-DAG: EntryPoint {{[0-9]+}} [[KERNEL0:[0-9]+]] "k_float_controls_0"
; SPV-DAG: EntryPoint {{[0-9]+}} [[KERNEL1:[0-9]+]] "k_float_controls_1"
; SPV-DAG: EntryPoint {{[0-9]+}} [[KERNEL2:[0-9]+]] "k_float_controls_2"
; SPV-DAG: EntryPoint {{[0-9]+}} [[KERNEL3:[0-9]+]] "k_float_controls_3"
; SPV-DAG: EntryPoint {{[0-9]+}} [[KERNEL4:[0-9]+]] "k_float_controls_4"
!0 = !{ptr @k_float_controls_0, !"k_float_controls_0", !1, i32 0, !2, !3, !4, i32 0, i32 0}
!1 = !{i32 2, i32 2}
!2 = !{i32 32, i32 36}
!3 = !{i32 0, i32 0}
!4 = !{!"", !""}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 8.0.1"}
!14 = !{i32 1, i32 0}

; SPV-DAG: ExecutionMode [[KERNEL0]] 4459 64
!15 = !{ptr @k_float_controls_0, i32 4459, i32 64}
; SPV-DAG: ExecutionMode [[KERNEL0]] 4459 32
!16 = !{ptr @k_float_controls_0, i32 4459, i32 32}
; SPV-DAG: ExecutionMode [[KERNEL0]] 4459 16
!17 = !{ptr @k_float_controls_0, i32 4459, i32 16}

; SPV-DAG: ExecutionMode [[KERNEL1]] 4460 64
!18 = !{ptr @k_float_controls_1, i32 4460, i32 64}
; SPV-DAG: ExecutionMode [[KERNEL1]] 4460 32
!19 = !{ptr @k_float_controls_1, i32 4460, i32 32}
; SPV-DAG: ExecutionMode [[KERNEL1]] 4460 16
!20 = !{ptr @k_float_controls_1, i32 4460, i32 16}

; SPV-DAG: ExecutionMode [[KERNEL2]] 4461 64
!21 = !{ptr @k_float_controls_2, i32 4461, i32 64}
; SPV-DAG: ExecutionMode [[KERNEL2]] 4461 32
!22 = !{ptr @k_float_controls_2, i32 4461, i32 32}
; SPV-DAG: ExecutionMode [[KERNEL2]] 4461 16
!23 = !{ptr @k_float_controls_2, i32 4461, i32 16}

; SPV-DAG: ExecutionMode [[KERNEL3]] 4462 64
!24 = !{ptr @k_float_controls_3, i32 4462, i32 64}
; SPV-DAG: ExecutionMode [[KERNEL3]] 4462 32
!25 = !{ptr @k_float_controls_3, i32 4462, i32 32}
; SPV-DAG: ExecutionMode [[KERNEL3]] 4462 16
!26 = !{ptr @k_float_controls_3, i32 4462, i32 16}

; SPV-DAG: ExecutionMode [[KERNEL4]] 4463 64
!27 = !{ptr @k_float_controls_4, i32 4463, i32 64}
; SPV-DAG: ExecutionMode [[KERNEL4]] 4463 32
!28 = !{ptr @k_float_controls_4, i32 4463, i32 32}
; SPV-DAG: ExecutionMode [[KERNEL4]] 4463 16
!29 = !{ptr @k_float_controls_4, i32 4463, i32 16}
