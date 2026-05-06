; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_KHR_abort
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; FIXME: enable the following run when the translator CI is updated to a new
; version of the SPIR-V Tools that includes the support for the SPV_KHR_abort
; extension.
; TODO: RUNx: spirv-val %t.spv

; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_KHR_abort

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: Capability AbortKHR
; CHECK-SPIRV: Extension "SPV_KHR_abort"
; CHECK-SPIRV: TypeInt [[#I32Ty:]] 32
; CHECK-SPIRV: FunctionParameter [[#I32Ty]] [[#MsgId:]]
; CHECK-SPIRV: AbortKHR [[#I32Ty]] [[#MsgId]]

; CHECK-LLVM: call spir_func void @{{.*__spirv_AbortKHR.*}}(i32 %{{.*}}){{.*}}#[[#ATTR:]]
; CHECK-LLVM-NEXT: unreachable
; CHECK-LLVM: attributes #[[#ATTR]] = {{{.*}}noreturn{{.*}}}

define spir_func void @test_abort(i32 %msg) {
entry:
  call spir_func void @_Z16__spirv_AbortKHRIiEvT_(i32 %msg)
  ret void
}

; Same as @test_abort, but with an explicit `unreachable` terminator instead of
; `ret void`. Both forms must lower to a single OpAbortKHR with no trailing
; OpUnreachable / OpReturn.
define spir_func void @test_abort_unreachable(i32 %msg) {
entry:
  call spir_func void @_Z16__spirv_AbortKHRIiEvT_(i32 %msg)
  unreachable
}

declare spir_func void @_Z16__spirv_AbortKHRIiEvT_(i32)

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
