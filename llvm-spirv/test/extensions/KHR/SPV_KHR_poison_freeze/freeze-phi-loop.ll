; RUN: llvm-spirv %s -o %t.noext.spv
; RUN: llvm-spirv %t.noext.spv -to-text -o %t.noext.spt
; RUN: FileCheck < %t.noext.spt %s --check-prefix=CHECK-NOEXT
;
; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_KHR_poison_freeze
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-NOEXT-NOT: Capability PoisonFreezeKHR
; CHECK-NOEXT-NOT: Extension "SPV_KHR_poison_freeze"
; CHECK-NOEXT-NOT: FreezeKHR

; CHECK-SPIRV: Capability PoisonFreezeKHR
; CHECK-SPIRV: Extension "SPV_KHR_poison_freeze"
; CHECK-SPIRV: FreezeKHR

define spir_kernel void @kernel_freeze_phi(i1 %in) {
entry:
  br label %loop

loop:
  %acc = phi i1 [ %fr, %loop ], [ false, %entry ]
  %fr = freeze i1 %in
  br label %loop
}

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
