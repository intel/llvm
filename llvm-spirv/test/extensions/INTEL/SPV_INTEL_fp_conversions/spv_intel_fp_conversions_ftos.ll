; Test that only the FloatConversionsFtoSINTEL capability is declared when
; the module uses only OpClampConvertFToSINTEL / OpClampStochasticRoundFToSINTEL.

; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_INTEL_int4,+SPV_INTEL_fp_conversions
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV --implicit-check-not="Capability FloatConversionsFtoFINTEL"
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Capability FloatConversionsFtoSINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_fp_conversions"

; CHECK-SPIRV: ClampConvertFToSINTEL
; CHECK-SPIRV: ClampStochasticRoundFToSINTEL

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-LLVM-LABEL: hf16_int4_clamp
; CHECK-LLVM: %[[#Call:]] = call spir_func i4 @_Z43__builtin_spirv_ClampConvertFP16ToInt4INTELDh(half 1.000000e+00)
define spir_func i4 @hf16_int4_clamp() {
entry:
  %0 = call spir_func i4 @_Z43__builtin_spirv_ClampConvertFP16ToInt4INTELDh(half 1.0)
  ret i4 %0
}

declare dso_local spir_func i4 @_Z43__builtin_spirv_ClampConvertFP16ToInt4INTELDh(half)

; CHECK-LLVM-LABEL: hf16_int4_stochastic
; CHECK-LLVM: %[[#Call:]] = call spir_func i4 @_Z51__builtin_spirv_ClampStochasticRoundFP16ToInt4INTELDhi(half 1.000000e+00, i32 1)
define spir_func i4 @hf16_int4_stochastic() {
entry:
  %0 = call spir_func i4 @_Z51__builtin_spirv_ClampStochasticRoundFP16ToInt4INTELhs(half 1.0, i32 1)
  ret i4 %0
}

declare dso_local spir_func i4 @_Z51__builtin_spirv_ClampStochasticRoundFP16ToInt4INTELhs(half, i32)
