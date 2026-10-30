; OpClampStochasticRoundFToSINTEL accepts no explicit FPRoundingMode.

; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_INTEL_int4,+SPV_INTEL_fp_conversions
; RUN: not llvm-spirv %t.spv -r -o %t.rev.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: FPRoundingMode: not supported on this conversion.

target triple = "spir-unknown-unknown"

declare dso_local spir_func i4 @_Z51__builtin_spirv_ClampStochasticRoundFP16ToInt4INTELDhi(half, i32)

define spir_func i4 @f() {
entry:
  %0 = call spir_func i4 @_Z51__builtin_spirv_ClampStochasticRoundFP16ToInt4INTELDhi(half 1.0, i32 1), !spirv.Decorations !1
  ret i4 %0
}

!1 = !{!2}
!2 = !{i32 39, i32 0}
