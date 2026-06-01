; REQUIRES: spirv-dis
; RUN: llvm-spirv %s -o %t.spv
; RUN: spirv-dis %t.spv | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64-unknown-unknown"

; Test that addrspacecast of generic null pointer generates GenericCastToPtr
; instruction in SPIR-V.

; CHECK: %_ptr_Workgroup_uchar = OpTypePointer Workgroup %uchar
; CHECK: %[[NULL:[0-9]+]] = OpConstantNull %_ptr_Generic_uchar
; CHECK: %[[CAST:[0-9]+]] = OpGenericCastToPtr %_ptr_Workgroup_uchar %[[NULL]]
; CHECK: OpPtrEqual %bool %[[CAST]]

define spir_kernel void @bar_generic_null(ptr addrspace(3) %arg) {
  %expr = icmp eq ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)), %arg
  ret void
}
