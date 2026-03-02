; RUN: llvm-spirv --spirv-ext=+SPV_KHR_float_controls2 -spirv-text %s -o - | FileCheck %s --check-prefix=SPIRV
; RUN: llvm-spirv --spirv-ext=+SPV_KHR_float_controls2 %s -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis | FileCheck %s --check-prefix=IR

; Verifies translation of contract fast-math on vectors types and preservation
; during round-trip back to LLVM-IR.

; SPIRV-DAG: TypeFloat [[#half:]] 16
; SPIRV-DAG: TypeFloat [[#float:]] 32
; SPIRV-DAG: TypeFloat [[#double:]] 64
; SPIRV-DAG: TypeVector [[#vec_half:]] [[#half]] 2
; SPIRV-DAG: TypeVector [[#vec_float:]] [[#float]] 2
; SPIRV-DAG: TypeVector [[#vec_double:]] [[#double]] 2
; SPIRV-DAG: FAdd [[#vec_half]] [[#rh:]]
; SPIRV-DAG: FAdd [[#vec_float]] [[#rf:]]
; SPIRV-DAG: FAdd [[#vec_double]] [[#rd:]]
; SPIRV-DAG: Decorate [[#rh]] FPFastMathMode 65536
; SPIRV-DAG: Decorate [[#rf]] FPFastMathMode 65536
; SPIRV-DAG: Decorate [[#rd]] FPFastMathMode 65536

target triple = "spirv-unknown-unknown"

define spir_kernel void @foo(<2 x half> %ah, <2 x half> %bh, <2 x float> %af, <2 x float> %bf, <2 x double> %ad, <2 x double> %bd) {
entry:
  ; IR-LABEL: define {{.*}} @foo
  ; IR-NEXT: entry:
  ; IR-NEXT:   %rh = fadd contract <2 x half> %ah, %bh
  ; IR-NEXT:   %rf = fadd contract <2 x float> %af, %bf
  ; IR-NEXT:   %rd = fadd contract <2 x double> %ad, %bd
  %rh = fadd contract <2 x half> %ah, %bh
  %rf = fadd contract <2 x float> %af, %bf
  %rd = fadd contract <2 x double> %ad, %bd
  ret void
}
