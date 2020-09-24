; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv --spirv-ext=+SPV_KHR_float_controls,+SPV_INTEL_float_controls2 %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis %t.bc -o %t.ll
; RUN: FileCheck %s --input-file %t.spt -check-prefix=SPV
; RUN: FileCheck %s --input-file %t.ll  -check-prefix=LLVM

; ModuleID = 'float_control_empty.bc'
source_filename = "float_control_empty.cpp"
target datalayout = "e-p:64:64-i64:64-n8:16:32"
target triple = "spir"

; LLVM-NOT: VCFloatControl
; SPV-NOT: {{ExecutionMode.*(4459|4460|4461|4462|4463|5620|5621|5622|5623)}}
; Function Attrs: noinline norecurse nounwind readnone
define dso_local dllexport spir_kernel void @k_no_fc(i32 %ibuf, i32 %obuf) local_unnamed_addr #16 {
entry:
  ret void
}

attributes #16 = { noinline norecurse nounwind readnone "VCMain" "VCFunction" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 8.0.1"}
