; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_KHR_poison_freeze
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; Per SPV_KHR_poison_freeze, declaring the PoisonFreezeKHR capability requires
; every entry point to use the ArithmeticPoisonKHR execution mode.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: Capability PoisonFreezeKHR
; CHECK-SPIRV: Extension "SPV_KHR_poison_freeze"
; CHECK-SPIRV-DAG: EntryPoint [[#]] [[#KFreeze:]] "kernel_uses_freeze"
; CHECK-SPIRV-DAG: EntryPoint [[#]] [[#KPoison:]] "kernel_uses_poison"
; CHECK-SPIRV-DAG: EntryPoint [[#]] [[#KPlain:]] "kernel_plain"
; CHECK-SPIRV-NOT: EntryPoint [[#]] [[#Helper:]] "helper"
; ArithmeticPoisonKHR (5157) must be present on every entry point, including
; the one that does not itself emit any poison/freeze op. Non-entry-point
; functions must not receive it.
; CHECK-SPIRV-DAG: ExecutionMode [[#KFreeze]] 5157
; CHECK-SPIRV-DAG: ExecutionMode [[#KPoison]] 5157
; CHECK-SPIRV-DAG: ExecutionMode [[#KPlain]] 5157
; CHECK-SPIRV-NOT: ExecutionMode [[#Helper:]] 5157{{$}}

; The reverse-translated metadata must mention every kernel.
; CHECK-LLVM-DAG: !{ptr @kernel_uses_freeze, i32 5157}
; CHECK-LLVM-DAG: !{ptr @kernel_uses_poison, i32 5157}
; CHECK-LLVM-DAG: !{ptr @kernel_plain, i32 5157}
; CHECK-LLVM-NOT: !{ptr @helper, i32 5157}

define spir_func i32 @helper(i32 %x) {
entry:
  %frozen = freeze i32 %x
  ret i32 %frozen
}

define spir_kernel void @kernel_uses_freeze(i32 %x, ptr addrspace(1) %out) {
entry:
  %frozen = freeze i32 %x
  store i32 %frozen, ptr addrspace(1) %out
  ret void
}

define spir_kernel void @kernel_uses_poison(ptr addrspace(1) %out) {
entry:
  store i32 poison, ptr addrspace(1) %out
  ret void
}

define spir_kernel void @kernel_plain(ptr addrspace(1) %out) {
entry:
  %r = call i32 @helper(i32 7)
  store i32 %r, ptr addrspace(1) %out
  ret void
}

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
