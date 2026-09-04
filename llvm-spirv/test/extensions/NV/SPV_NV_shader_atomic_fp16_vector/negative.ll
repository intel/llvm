; RUN: llvm-as < %s -o %t.bc
; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s

; Without SPV_NV_shader_atomic_fp16_vector an fp16 vector atomic cannot be
; translated.

; CHECK: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-NEXT: SPV_NV_shader_atomic_fp16_vector

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

define spir_func void @test(ptr addrspace(1) %p, <2 x half> %v2) {
entry:
  %a = atomicrmw fadd ptr addrspace(1) %p, <2 x half> %v2 seq_cst
  ret void
}
