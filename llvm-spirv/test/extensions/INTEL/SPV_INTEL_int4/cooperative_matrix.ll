; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_INTEL_int4 -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Capability Int4TypeINTEL
; CHECK-SPIRV-DAG: Capability CooperativeMatrixKHR
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_int4"
; CHECK-SPIRV-DAG: Capability Int4CooperativeMatrixINTEL
; CHECK-SPIRV-DAG: Extension "SPV_KHR_cooperative_matrix"
; CHECK-SPIRV-DAG: TypeInt [[#Int4Ty:]] 4 0
; CHECK-SPIRV-DAG: TypeCooperativeMatrixKHR [[#CoopMatTy:]] [[#Int4Ty]] [[#]] [[#]] [[#]] [[#]]
; CHECK-SPIRV-DAG: CompositeConstruct [[#CoopMatTy]]

; CHECK-LLVM: call spir_func target("spirv.CooperativeMatrixKHR", i4, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructi(i4 0)

; ModuleID = 'matrix-int4-test.bc'
source_filename = "matrix-int4-test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_kernel void @foo() {
entry:
  %call.i.i = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i4, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(i4 noundef 0)
  ret void
}

declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i4, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(i4 noundef)
