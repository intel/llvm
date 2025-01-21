; RUN: llvm-as < %s -o %t.bc
; RUN: not llvm-spirv %t.bc --spirv-ext=+SPV_KHR_cooperative_matrix -o %t.spv 2>&1 | FileCheck %s

; CHECK: InvalidInstruction: Can't translate llvm instruction:
; CHECK: TypeCooperativeMatrixKHR
; CHECK: Unsupported Scope parameter

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define void @convert_f_to_u() {
entry:
  %0 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 8, 12, 12, 2) @_Z26__spirv_CompositeConstructFloat(float 0.000000e+00)
  ret void
}

declare spir_func noundef target("spirv.CooperativeMatrixKHR", float, 8, 12, 12, 2) @_Z26__spirv_CompositeConstructFloat(float noundef)
