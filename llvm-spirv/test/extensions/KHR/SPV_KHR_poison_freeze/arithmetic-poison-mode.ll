; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_KHR_poison_freeze
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %s -spirv-text -o %t.noext.spt
; RUN: FileCheck < %t.noext.spt %s --check-prefix=CHECK-NOEXT

; Test that we emit ExecutionModeArithmeticPoisonKHR exactly once.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: Capability PoisonFreezeKHR
; CHECK-SPIRV: Extension "SPV_KHR_poison_freeze"
; CHECK-SPIRV: EntryPoint [[#]] [[#Kernel:]] "test_kernel"
; CHECK-SPIRV-COUNT-1: ExecutionMode [[#Kernel]] 5157
; CHECK-SPIRV-NOT: ExecutionMode [[#Kernel]] 5157

; CHECK-NOEXT-NOT: Capability PoisonFreezeKHR
; CHECK-NOEXT-NOT: Extension "SPV_KHR_poison_freeze"
; CHECK-NOEXT-NOT: ExecutionMode {{[0-9]+}} 5157

; CHECK-LLVM: !spirv.ExecutionMode = !{![[#EM:]]{{.*}}}
; CHECK-LLVM: ![[#EM]] = !{ptr @test_kernel, i32 5157}

define spir_kernel void @test_kernel() {
entry:
  ret void
}

!spirv.ExecutionMode = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}

!0 = !{ptr @test_kernel, i32 5157}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
