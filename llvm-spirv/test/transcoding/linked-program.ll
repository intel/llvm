; REQUIRES: spirv-link
;
; RUN: llvm-spirv %s -o %t.spv
; RUN: spirv-val %t.spv
; RUN: spirv-link %t.spv -o %t.linked.spv
; RUN: llvm-spirv -r %t.linked.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s
; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv32-unknown-unknown -filetype=obj %s -o %t.llc.spv %}
; RUN: %if spirv-backend %{ spirv-link %t.llc.spv -o %t.llc.linked.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r %t.llc.linked.spv -o %t.llc.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.rev.bc -o %t.llc.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s < %t.llc.rev.ll %}
;
; This checks that SPIR-V programs with global variables are still consumable
; after spirv-link.

target triple = "spir-unknown-unknown"

@foo = common dso_local local_unnamed_addr addrspace(1) global i32 0, align 4
; CHECK: @foo = internal addrspace(1) global i32 0, align 4

define dso_local spir_kernel void @bar() local_unnamed_addr {
entry:
  store i32 42, ptr addrspace(1) @foo, align 4
  ret void
}
