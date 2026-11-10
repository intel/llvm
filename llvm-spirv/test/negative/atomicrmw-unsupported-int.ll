; Check that integer atomicrmw operations without a SPIR-V opcode are reported
; as unsupported instead of crashing the writer.
;
; The writer stops at the first unsupported instruction, so each operation is
; translated from its own copy of the module.

; RUN: sed -e 's/OP/usub_cond/' %s > %t.ll
; RUN: not llvm-spirv %t.ll -o %t.spv 2>&1 | FileCheck %s -DOP=usub_cond
; RUN: sed -e 's/OP/usub_sat/' %s > %t.ll
; RUN: not llvm-spirv %t.ll -o %t.spv 2>&1 | FileCheck %s -DOP=usub_sat

; CHECK: InvalidInstruction: Can't translate llvm instruction:
; CHECK: Atomic [[OP]] is not supported in SPIR-V!

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test(ptr addrspace(1) %p) {
  %r = atomicrmw OP ptr addrspace(1) %p, i32 1 monotonic
  ret void
}
