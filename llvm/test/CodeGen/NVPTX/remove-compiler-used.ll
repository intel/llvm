;; This test verifies llc on NVPTX will delete the llvm.compiler.used symbol
;; while keeping the symbol in the outputted ASM.

; RUN: llc < %s -march=nvptx64 | FileCheck %s

@keep_this = internal global i32 2, align 4
@llvm.compiler.used = appending global [1 x ptr] [ptr @keep_this], section "llvm.metadata"

; CHECK-NOT: llvm.metadata
; CHECK-NOT: llvm{{.*}}used
; CHECK-NOT: llvm{{.*}}compiler{{.*}}used

; CHECK: .global .align 4 .u32 keep_this

; CHECK-NOT: llvm.metadata
; CHECK-NOT: llvm{{.*}}used
; CHECK-NOT: llvm{{.*}}compiler{{.*}}used
