; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_KHR_poison_freeze
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %s -spirv-text -o %t.noext.spt
; RUN: FileCheck < %t.noext.spt %s --check-prefix=CHECK-NOEXT

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: Capability PoisonFreezeKHR
; CHECK-SPIRV: Extension "SPV_KHR_poison_freeze"
; CHECK-SPIRV: TypeInt [[#I32Ty:]] 32 0
; CHECK-SPIRV: TypeInt [[#I64Ty:]] 64 0
; CHECK-SPIRV-COUNT-1: PoisonKHR [[#I32Ty]] [[#PoisonI32:]]
; CHECK-SPIRV-COUNT-1: PoisonKHR [[#I64Ty]] [[#PoisonI64:]]
; CHECK-SPIRV: ReturnValue [[#PoisonI32]]
; CHECK-SPIRV: ReturnValue [[#PoisonI64]]
; CHECK-SPIRV: ReturnValue [[#PoisonI32]]

; CHECK-NOEXT-NOT: Capability PoisonFreezeKHR
; CHECK-NOEXT-NOT: Extension "SPV_KHR_poison_freeze"
; CHECK-NOEXT-NOT: PoisonKHR
; CHECK-NOEXT: Undef

; CHECK-LLVM: ret i32 poison
; CHECK-LLVM: ret i64 poison
; CHECK-LLVM: ret i32 poison

define spir_func i32 @test_poison_i32() {
entry:
  ret i32 poison
}

define spir_func i64 @test_poison_i64() {
entry:
  ret i64 poison
}

define spir_func i32 @test_poison_i32_again() {
entry:
  ret i32 poison
}

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
