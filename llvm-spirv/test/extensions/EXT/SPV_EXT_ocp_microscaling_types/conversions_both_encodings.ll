; Checks that both the EXT builtins (SPV_EXT_ocp_microscaling_types,
; Float4E2M1EXT encoding 4225) and the INTEL builtins (SPV_INTEL_float4,
; Float4E2M1INTEL encoding 6214) can coexist in one module with both
; extensions enabled, and that each conversion round-trips back to its own
; builtin.

; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_INTEL_float4,+SPV_EXT_ocp_microscaling_types,+SPV_INTEL_int4
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Capability Float4EXT
; CHECK-SPIRV-DAG: Capability Float4E2M1INTEL
; CHECK-SPIRV-DAG: Extension "SPV_EXT_ocp_microscaling_types"
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_float4"

; CHECK-SPIRV-DAG: Name [[#ext_conv:]] "ext_conv"
; CHECK-SPIRV-DAG: Name [[#intel_conv:]] "intel_conv"

; CHECK-SPIRV-DAG: TypeFloat [[#E2M1ExtTy:]] 4 4225
; CHECK-SPIRV-DAG: TypeFloat [[#E2M1IntelTy:]] 4 6214

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-SPIRV: Function [[#]] [[#ext_conv]] [[#]]
; CHECK-SPIRV: Bitcast [[#E2M1ExtTy]] [[#ExtCast:]] [[#]]
; CHECK-SPIRV: FConvert [[#]] [[#]] [[#ExtCast]]

; CHECK-LLVM-LABEL: ext_conv
; CHECK-LLVM: call spir_func half @_Z36__builtin_spirv_ConvertE2M1ToFP16EXTi(i4 1)

define spir_func half @ext_conv() {
entry:
  %r = call spir_func half @_Z36__builtin_spirv_ConvertE2M1ToFP16EXTi(i4 1)
  ret half %r
}

declare dso_local spir_func half @_Z36__builtin_spirv_ConvertE2M1ToFP16EXTi(i4)

; CHECK-SPIRV: Function [[#]] [[#intel_conv]] [[#]]
; CHECK-SPIRV: Bitcast [[#E2M1IntelTy]] [[#IntelCast:]] [[#]]
; CHECK-SPIRV: FConvert [[#]] [[#]] [[#IntelCast]]

; CHECK-LLVM-LABEL: intel_conv
; CHECK-LLVM: call spir_func half @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELi(i4 1)

define spir_func half @intel_conv() {
entry:
  %r = call spir_func half @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELi(i4 1)
  ret half %r
}

declare dso_local spir_func half @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELi(i4)
