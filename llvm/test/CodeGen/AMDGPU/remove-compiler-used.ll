;; This test verifies llc on AMDGCN will delete the llvm.compiler.used symbol
;; while keeping the symbol in the outputted ASM.

; RUN: llc < %s -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck %s
; RUN: llc < %s -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 | FileCheck %s
; RUN: llc < %s -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a | FileCheck %s

@keep_this = internal global i32 2, align 4
@llvm.compiler.used = appending global [1 x ptr] [ptr @keep_this], section "llvm.metadata"

; CHECK-NOT: llvm.metadata
; CHECK-NOT: llvm{{.*}}used
; CHECK-NOT: llvm{{.*}}compiler{{.*}}used

; CHECK: .type keep_this,@object ;

; CHECK-NOT: llvm.metadata
; CHECK-NOT: llvm{{.*}}used
; CHECK-NOT: llvm{{.*}}compiler{{.*}}used
