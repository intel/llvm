; RUN: llvm-spirv --spirv-ext=+SPV_KHR_float_controls2 -spirv-text %s -o - | FileCheck %s --check-prefix=SPIRV
; RUN: llvm-spirv --spirv-ext=+SPV_KHR_float_controls2 %s -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis | FileCheck %s --check-prefix=IR

; By default, do not set the execution-mode when the extension is used.
; This test doesn't verify directly that the instructions have the SPIRV
; 'contract' flag (this is done in another test).
; As a sanity check, we still reverse-translate and check the IR.
; 6028 is FPFastMathDefault 
; SPIRV: Capability FloatControls2
; SPIRV: Extension "SPV_KHR_float_controls2"
; SPIRV-NOT: ExecutionModeId {{[0-9]+}} 6028

target triple = "spirv-unknown-unknown"

define spir_kernel void @foo(half %ah, half %bh, float %af, float %bf, double %ad, double %bd) {
entry:
  ; IR-LABEL: define {{.*}} @foo
  ; IR-NEXT: entry:
  ; IR-NEXT:   %{{.*}} = fadd contract half %{{.*}}, %{{.*}}
  ; IR-NEXT:   %{{.*}} = fadd contract float %{{.*}}, %{{.*}}
  ; IR-NEXT:   %{{.*}} = fadd contract double %{{.*}}, %{{.*}}
  ; IR-NEXT    call void @bar(half %rh, half %bh, float %rf, float %bf, double %rd, double %bd)
  %rh = fadd contract half %ah, %bh
  %rf = fadd contract float %af, %bf
  %rd = fadd contract double %ad, %bd
  call void @bar(half %rh, half %bh, float %rf, float %bf, double %rd, double %bd)
  ret void
}

define internal void @bar(half %ah, half %bh, float %af, float %bf, double %ad, double %bd) {
entry:
  ; IR-LABEL: define {{.*}} @bar
  ; IR-NEXT: entry:
  ; IR-NEXT:   %{{.*}} = fadd contract half %{{.*}}, %{{.*}}
  ; IR-NEXT:   %{{.*}} = fadd contract float %{{.*}}, %{{.*}}
  ; IR-NEXT:   %{{.*}} = fadd contract double %{{.*}}, %{{.*}}
  %rh = fadd contract half %ah, %bh
  %rf = fadd contract float %af, %bf
  %rd = fadd contract double %ad, %bd
  ret void
}
