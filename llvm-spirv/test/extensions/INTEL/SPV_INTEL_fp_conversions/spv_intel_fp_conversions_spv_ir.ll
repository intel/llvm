; Test for conversions, that don't require special type interpretation.

; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_INTEL_fp_conversions
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Capability FloatConversionsINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_fp_conversions"
; CHECK-SPIRV-DAG: TypeFloat [[#HalfTy:]] 16
; CHECK-SPIRV-DAG: TypeFloat [[#FloatTy:]] 32
; CHECK-SPIRV-DAG: TypeInt [[#ShortTy:]] 16 0
; CHECK-SPIRV-DAG: Constant [[#ShortTy]] [[#IntConst:]] 4
; CHECK-SPIRV-DAG: Constant [[#FloatTy]] [[#FPConst:]] 1065353216

; CHECK-SPIRV: ClampConvertFToFINTEL [[#HalfTy]] [[#]] [[#FPConst]]
; CHECK-SPIRV: StochasticRoundFToFINTEL [[#HalfTy]] [[#]] [[#FPConst]] [[#IntConst]]
; CHECK-SPIRV: ClampStochasticRoundFToFINTEL [[#HalfTy]] [[#]] [[#FPConst]] [[#IntConst]]
; CHECK-SPIRV: ClampStochasticRoundFToFINTEL [[#HalfTy]] [[#]] [[#FPConst]] [[#IntConst]] [[#]]

; CHECK-LLVM: call spir_func half @_Z35__spirv_ClampConvertFToFINTEL_Rhalff(float 1.000000e+00)
; CHECK-LLVM: call spir_func half @_Z38__spirv_StochasticRoundFToFINTEL_Rhalffs(float 1.000000e+00, i16 4)
; CHECK-LLVM: call spir_func half @_Z43__spirv_ClampStochasticRoundFToFINTEL_Rhalffs(float 1.000000e+00, i16 4)
; CHECK-LLVM: call spir_func half @_Z43__spirv_ClampStochasticRoundFToFINTEL_RhalffsPs(float 1.000000e+00, i16 4, ptr null)

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind readnone
define spir_func void @foo() {
entry:
  %0 = call spir_func half @_Z29__spirv_ClampConvertFToFINTELf(float 1.0)
  %2 = call spir_func half @_Z32__spirv_StochasticRoundFToFINTELfs(float 1.0, i16 4)
  %3 = call spir_func half @_Z37__spirv_ClampStochasticRoundFToFINTELfs(float 1.0, i16 4)
  %5 = call spir_func half @_Z37__spirv_ClampStochasticRoundFToFINTELfsPs(float 1.0, i16 4, ptr null)
  ret void
}

declare dso_local spir_func half @_Z29__spirv_ClampConvertFToFINTELf(float)

declare dso_local spir_func half @_Z32__spirv_StochasticRoundFToFINTELfs(float, i16)

declare dso_local spir_func half @_Z37__spirv_ClampStochasticRoundFToFINTELfs(float, i16)

declare dso_local spir_func half @_Z37__spirv_ClampStochasticRoundFToFINTELfsPs(float, i16, ptr)
