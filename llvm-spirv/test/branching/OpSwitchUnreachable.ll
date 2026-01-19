; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

define void @test_switch_with_unreachable_block(i1 %a) {
  %value = zext i1 %a to i32
; CHECK-SPIRV:      Switch [[#]] [[#UNREACHABLE:]] 0 [[#REACHABLE:]] 1 [[#REACHABLE:]]
; CHECK-LLVM: switch i32 %{{.*}}, label %[[unreachable:]]
; CHECK-LLVM:   i32 0, label %[[reachable:]]
; CHECK-LLVM:   i32 1, label %[[reachable]]

  switch i32 %value, label %unreachable [
    i32 0, label %reachable
    i32 1, label %reachable
  ]

reachable:
  ret void

; CHECK-SPIRV: Label [[#REACHABLE]]
; CHECK-SPIRV: Return
; CHECK-LLVM: [[reachable]]:
; CHECK-LLVM:   ret void

unreachable:
  unreachable

; CHECK-SPIRV: Label [[#UNREACHABLE]]
; CHECK-SPIRV: Unreachable
; CHECK-LLVM: [[unreachable]]:
; CHECK-LLVM:   unreachable
}
