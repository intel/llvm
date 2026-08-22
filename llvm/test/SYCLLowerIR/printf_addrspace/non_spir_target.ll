;; Verify that the pass does NOT set spir_func calling convention on the
;; printf declaration when the target triple is not SPIR/SPIR-V.

; RUN: opt < %s -passes=SYCLMutatePrintfAddrspace -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [6 x i8] c"hello\00", align 1

define void @test_printf() {
entry:
  %call = call i32 @_Z18__spirv_ocl_printfIJEEiPKcDpT_(ptr @.str)
  ret void
}

; CHECK: call i32 (ptr addrspace(2), ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2)
; The declaration should NOT have spir_func CC for non-SPIR targets.
; CHECK: declare dso_local i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2), ...)
; CHECK-NOT: declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz

declare i32 @_Z18__spirv_ocl_printfIJEEiPKcDpT_(ptr)
