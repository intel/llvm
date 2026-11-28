; OpClampConvertFToSINTEL accepts no explicit FPRoundingMode.

; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_INTEL_int4,+SPV_INTEL_fp_conversions
; RUN: not llvm-spirv %t.spv -r -o %t.rev.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: FPRoundingMode: not supported on this conversion.

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

define spir_func i4 @hf16_int4_clamp_rte() {
entry:
  %0 = call spir_func i4 @_Z43__builtin_spirv_ClampConvertFP16ToInt4INTELDh(half 1.0), !spirv.Decorations !1
  ret i4 %0
}

declare dso_local spir_func i4 @_Z43__builtin_spirv_ClampConvertFP16ToInt4INTELDh(half)

!1 = !{!2}
!2 = !{i32 39, i32 0}
