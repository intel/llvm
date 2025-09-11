; Check that we don't end up with duplicated array types in TypeMap.
; No FileCheck needed, we only want to check the absence of errors.
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; ModuleID = 'duplicate-array-types'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%duplicate = type { [2 x ptr addrspace(4)] }

; Function Attrs: mustprogress norecurse nounwind
define spir_kernel void @foo() {
entry:
  alloca [2 x ptr addrspace(4)], align 8
  alloca %duplicate, align 8
  ret void
}
