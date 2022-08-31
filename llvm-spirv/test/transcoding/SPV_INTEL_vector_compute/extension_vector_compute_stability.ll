; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_vector_compute,+SPV_KHR_float_controls,+SPV_INTEL_float_controls2 --spirv-allow-unknown-intrinsics=llvm.genx
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_vector_compute,+SPV_KHR_float_controls,+SPV_INTEL_float_controls2 --spirv-allow-unknown-intrinsics
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis %t.bc -o %t.ll
; RUN: FileCheck %s --input-file %t.ll  -check-prefix=LLVM

; ModuleID = 'slm.bc'
source_filename = "slm.cpp"
target datalayout = "e-p:64:64-i64:64-n8:16:32"
target triple = "spir"


; LLVM-DAG: @k_rte{{[^a-zA-Z0-9_][^#]*}}#[[K_RTE_ATTR:[0-9]+]]
; LLVM-DAG: attributes #[[K_RTE_ATTR]]{{[^0-9].*"VCFloatControl"="0".*"VCFunction".*"VCSLMSize"="256"}}
; LLVM-DAG: @in = internal global{{[^#]*}}#[[IN_ATTR:[0-9]+]]
; LLVM-DAG: attributes #[[IN_ATTR]]{{[^0-9].*"VCByteOffset"="1".*"VCGlobalVariable".*"VCVolatile"}}

@in = internal global <256 x i8> undef, align 256 #0
declare <256 x i8> @llvm.genx.vload(<256 x i8>* nonnull %aaa)

; Function Attrs: noinline norecurse nounwind readnone
define dso_local dllexport spir_kernel void @k_rte(i32 %ibuf, i32 %obuf) local_unnamed_addr #1 {
entry:
  %gload53 = tail call <256 x i8> @llvm.genx.vload(<256 x i8>* nonnull @in)
  ret void
}

attributes #0 = { "VCByteOffset"="1" "VCVolatile" "VCGlobalVariable"}
attributes #1 = { noinline norecurse nounwind readnone "VCFloatControl"="0" "VCFunction" "VCSLMSize"="256" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "oclrt"="1" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

; Note
!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 8.0.1"}
