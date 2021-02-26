; REQUIRES: spirv-link
;
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: spirv-link %t.spv -o %t.linked.spv
; RUN: llvm-spirv -r %t.linked.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s
;
; This checks that SPIR-V programs with global variables are still consumable
; after spirv-link.

target triple = "spir-unknown-unknown"

@foo = common dso_local local_unnamed_addr addrspace(1) global i32 0, align 4
; CHECK: @foo = internal addrspace(1) global i32 0, align 4

define dso_local spir_kernel void @bar() local_unnamed_addr {
entry:
  store i32 42, i32 addrspace(1)* @foo, align 4
  ret void
}
