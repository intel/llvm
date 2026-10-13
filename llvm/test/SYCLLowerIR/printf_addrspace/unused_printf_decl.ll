;; Verify that the pass does NOT insert an AS2 printf declaration when
;; the module has a matching printf declaration with no call users, and
;; that the unused declaration itself is removed from the module.

; RUN: opt < %s -passes=SYCLMutatePrintfAddrspace -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_kernel void @kernel() {
entry:
  ret void
}

; This declaration matches the printf name prefix but has no users.
declare dso_local spir_func i32 @_Z18__spirv_ocl_printfIJfEEiPKcDpT_(ptr addrspace(4), float)

; No AS2 printf declaration should be created.
; CHECK-NOT: @_Z18__spirv_ocl_printfPU3AS2Kcz
; The unused generic-AS printf declaration should be removed.
; CHECK-NOT: @_Z18__spirv_ocl_printfIJfEEiPKcDpT_
