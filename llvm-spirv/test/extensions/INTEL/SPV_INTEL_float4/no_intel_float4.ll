; Checks that an *INTEL FP4 conversion builtin fails to translate when
; SPV_INTEL_float4 is not enabled.

; RUN: not llvm-spirv %s -o %t.spv --spirv-ext=+SPV_INTEL_int4 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_INTEL_float4

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

define spir_func half @intel_conv() {
entry:
  %r = call spir_func half @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELi(i4 1)
  ret half %r
}

declare dso_local spir_func half @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELi(i4)
