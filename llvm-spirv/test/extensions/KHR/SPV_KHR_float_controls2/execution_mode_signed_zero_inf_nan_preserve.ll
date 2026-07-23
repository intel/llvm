; RUN: llvm-spirv %s -o - -spirv-max-version=1.4 --spirv-ext=-SPV_KHR_float_controls2 --spirv-text | FileCheck %s --check-prefixes=FLOAT_CONTROLS,FLOAT_CONTROLS-OFF
; RUN: llvm-spirv %s -o - --spirv-ext=+SPV_KHR_float_controls2 --spirv-text | FileCheck %s --check-prefixes=FLOAT_CONTROLS,FLOAT_CONTROLS-ON
; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_KHR_float_controls2
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis | FileCheck %s --check-prefix=IR

; Check that we do not lower to SignedZeroInfNanPreserve when the FloatControls2 extension is available.
; Instead, we set FPFastMathDefault to 0.
;
; FLOAT_CONTROLS-OFF-NOT: Capability FloatControls2
; FLOAT_CONTROLS-ON-NOT: Capability SignedZeroInfNanPreserve
; FLOAT_CONTROLS-OFF: Capability SignedZeroInfNanPreserve
; FLOAT_CONTROLS-ON: Capability FloatControls2
; FLOAT_CONTROLS: EntryPoint {{[0-9]+}} [[#FOO:]] "foo"
; FLOAT_CONTROLS-OFF: ExecutionMode [[#FOO]] 4461 32
; FLOAT_CONTROLS-ON: ExecutionModeId [[#FOO]] 6028 [[#FLOAT:]] [[#ZERO:]]
; FLOAT_CONTROLS-ON-DAG: TypeFloat [[#FLOAT]] 32
; FLOAT_CONTROLS-ON-DAG: Constant {{[0-9]+}} [[#ZERO]] 0

target triple = "spirv-unknown-unknown"

define spir_kernel void @foo(half %ah, half %bh, float %af, float %bf, double %ad, double %bd) {
entry:
  ; IR-LABEL: define {{.*}} @foo
  ; IR-NEXT: entry:
  ; IR-NEXT: %{{.*}} = fadd contract half %{{.*}}, %{{.*}}
  ; IR-NEXT: %{{.*}} = fadd contract float %{{.*}}, %{{.*}}
  ; IR-NEXT: %{{.*}} = fadd contract double %{{.*}}, %{{.*}}
  %rh = fadd contract half %ah, %bh
  %rf = fadd contract float %af, %bf
  %rd = fadd contract double %ad, %bd
  ret void
}

 !spirv.ExecutionMode = !{!0}

 !0 = !{ptr @foo, i32 4461, i32 32}
