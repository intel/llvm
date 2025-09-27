; REQUIRES: spirv-dis
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-dis %t.spv | FileCheck %s

; Test that addrspacecast of null pointer generates appropriate OpConstantNull
; instruction in SPIR-V.

; CHECK: %_ptr_CrossWorkgroup_uchar = OpTypePointer CrossWorkgroup %uchar
; CHECK: %[[NULL:[0-9]+]] = OpConstantNull %_ptr_CrossWorkgroup_uchar
; We don't support OpPtrEqual instruction after KhronosGroup/SPIRV-LLVM-Translator@5e0f14b5a09d961 was reverted.
; TODO: use community check-lines after the change relands.
; CHECK: %[[U:[0-9]+]] = OpConvertPtrToU %ulong %[[NULL]]
; CHECK: OpIEqual %bool %[[U]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64-unknown-unknown"

define spir_kernel void @bar(ptr addrspace(1) %arg) {
pass26:
  %expr = icmp eq ptr addrspace(1) addrspacecast (ptr null to ptr addrspace(1)), %arg
  ret void
}
