; Conditional abort — the assert() pattern where only some paths abort.
; Verifies that:
;   1. The abort BB gets OpAbortKHR (not OpUnreachable)
;   2. The non-abort BB is unaffected (still has OpReturn)
;   3. No double terminators in the abort block.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_abort -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; Round-trip
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; FIXME: enable the following run when the translator CI is updated to a new
; version of the SPIR-V Tools that includes the support for the SPV_KHR_abort
; extension.
; TODO: RUNx: spirv-val %t.spv

; ---- SPIR-V with extension ----
; CHECK-SPIRV-DAG: Extension "SPV_KHR_abort"
; CHECK-SPIRV-DAG: Capability AbortKHR
; CHECK-SPIRV: Function
;
; Entry block: branch conditional
; CHECK-SPIRV: BranchConditional
;
; OK block: normal return
; CHECK-SPIRV: Return
;
; Trap block: abort only, no Unreachable / Return after it
; CHECK-SPIRV: AbortKHR
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: FunctionEnd

; ---- Round-trip LLVM IR ----
; CHECK-LLVM: define spir_func void @assert_like
; CHECK-LLVM: br i1
; CHECK-LLVM: ret void
; CHECK-LLVM: call spir_func void @{{.*__spirv_AbortKHR.*}}(i32 %{{.*}}){{.*}}#[[#ATTR:]]
; CHECK-LLVM-NEXT: unreachable
; CHECK-LLVM: attributes #[[#ATTR]] = {{{.*}}noreturn{{.*}}}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_func void @assert_like(i32 %gid, i32 %N, i32 %msg) {
entry:
  %cmp = icmp slt i32 %gid, %N
  br i1 %cmp, label %ok, label %trap

ok:
  ret void

trap:
  call spir_func void @_Z16__spirv_AbortKHRIiEvT_(i32 %msg)
  unreachable
}

declare spir_func void @_Z16__spirv_AbortKHRIiEvT_(i32)

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
