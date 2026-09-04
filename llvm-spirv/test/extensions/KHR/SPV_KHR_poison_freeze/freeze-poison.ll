; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_KHR_poison_freeze
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %s -o %t.noext.spv
; RUN: llvm-spirv %t.noext.spv -o %t.noext.spt --to-text
; RUN: FileCheck < %t.noext.spt %s --check-prefix=CHECK-NOEXT
; RUN: llvm-spirv %t.noext.spv -o %t.noext.rev.bc -r
; RUN: llvm-dis %t.noext.rev.bc -o %t.noext.rev.ll
; RUN: FileCheck < %t.noext.rev.ll %s --check-prefix=CHECK-NOEXT-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: Capability PoisonFreezeKHR
; CHECK-SPIRV: Extension "SPV_KHR_poison_freeze"
; CHECK-SPIRV: TypeInt [[#I32Ty:]] 32
; CHECK-SPIRV: PoisonKHR [[#I32Ty]] [[#PoisonId:]]
; CHECK-SPIRV: FreezeKHR [[#I32Ty]] [[#FrozenId:]] [[#PoisonId]]
; CHECK-SPIRV: ReturnValue [[#FrozenId]]

; CHECK-NOEXT-NOT: Capability PoisonFreezeKHR
; CHECK-NOEXT-NOT: Extension "SPV_KHR_poison_freeze"
; CHECK-NOEXT-NOT: PoisonKHR
; CHECK-NOEXT-NOT: FreezeKHR

; CHECK-LLVM: %[[FROZEN:[a-zA-Z0-9_.]+]] = freeze i32 poison
; CHECK-LLVM: ret i32 %[[FROZEN]]

; Without the extension, the freeze is stripped and replaced with i32 0.
; CHECK-NOEXT-LLVM: ret i32 0

define spir_func i32 @test_freeze_poison() {
entry:
  %frozen = freeze i32 poison
  ret i32 %frozen
}

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
