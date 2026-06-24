; Sanity check: enabling SPV_KHR_abort does not affect functions that don't
; call __spirv_AbortKHR. Normal terminators (OpReturn, OpReturnValue,
; OpBranch, OpUnreachable) must be preserved.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_abort -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV --implicit-check-not AbortKHR

; Validate SPIR-V
; RUN: spirv-val %t.spv

; Round-trip must be lossless for non-abort code
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; Normal void return preserved
; CHECK-SPIRV: Function
; CHECK-SPIRV: Return
; CHECK-SPIRV: FunctionEnd

; Normal value return preserved
; CHECK-SPIRV: Function
; CHECK-SPIRV: ReturnValue
; CHECK-SPIRV: FunctionEnd

; Unreachable preserved (not suppressed when no abort precedes it)
; CHECK-SPIRV: Function
; CHECK-SPIRV: Unreachable
; CHECK-SPIRV: FunctionEnd

; Branch preserved
; CHECK-SPIRV: Function
; CHECK-SPIRV: BranchConditional
; CHECK-SPIRV: ReturnValue
; CHECK-SPIRV: ReturnValue
; CHECK-SPIRV: FunctionEnd

; ---- Round-trip ----
; CHECK-LLVM: define spir_func void @void_return
; CHECK-LLVM: ret void
;
; CHECK-LLVM: define spir_func i32 @value_return
; CHECK-LLVM: ret i32
;
; CHECK-LLVM: define spir_func void @plain_unreachable
; CHECK-LLVM: unreachable
;
; CHECK-LLVM: define spir_func i32 @branched

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_func void @void_return() {
entry:
  ret void
}

define spir_func i32 @value_return(i32 %x) {
entry:
  %r = add i32 %x, 1
  ret i32 %r
}

define spir_func void @plain_unreachable() {
entry:
  unreachable
}

define spir_func i32 @branched(i1 %cond, i32 %a, i32 %b) {
entry:
  br i1 %cond, label %t, label %f

t:
  ret i32 %a

f:
  ret i32 %b
}

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
