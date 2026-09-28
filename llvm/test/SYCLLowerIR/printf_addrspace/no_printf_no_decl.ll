;; Verify that the pass does NOT insert a printf declaration when the module
;; has no printf functions that need transformation.

; RUN: opt < %s -passes=SYCLMutatePrintfAddrspace -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_kernel void @no_printf_kernel() {
entry:
  ret void
}

; CHECK-NOT: @_Z18__spirv_ocl_printfPU3AS2Kcz
