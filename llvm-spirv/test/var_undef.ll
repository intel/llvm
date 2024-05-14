; Ensure that encoding of variable with undef initializer
; has correct wordcount

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -to-text %t.spv -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix CHECK-LLVM

; CHECK-SPIRV:Name [[BAR_VAR:[0-9]+]] "bar"
;; bar variable does not have optional initializer
;; word count must be 4
; CHECK-SPIRV:4 Variable [[#]] [[BAR_VAR]]

; CHECK-LLVM:@bar = internal addrspace(3) global %range undef, align 8

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%anon = type { %range }
%range = type { %array }
%array = type { [2 x i64] }

@foo = internal addrspace(3) global %anon undef, align 8

@bar = internal unnamed_addr addrspace(3) global %range undef, align 8
