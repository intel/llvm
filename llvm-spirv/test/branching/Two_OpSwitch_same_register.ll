; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test_two_switch_same_register(i32 %value) {
; CHECK-SPIRV: Switch [[#REGISTER:]] [[#DEFAULT1:]] 1 [[#CASE1:]] 0 [[#CASE2:]]
; CHECK-LLVM: switch i32 %[[value:[^, ]+]], label %[[default1:[^ ]+]]

  switch i32 %value, label %default1 [
    i32 1, label %case1
    i32 0, label %case2
  ]

case1:
  br label %default1

case2:
  br label %default1

default1:
  switch i32 %value, label %default2 [
    i32 0, label %case3
    i32 1, label %case4
  ]

case3:
  br label %default2

case4:
  br label %default2

default2:
  ret void

; CHECK-SPIRV:      Label [[#CASE1]]
; CHECK-SPIRV-NEXT: Branch [[#DEFAULT1]]

; CHECK-SPIRV:      Label [[#CASE2]]
; CHECK-SPIRV-NEXT: Branch [[#DEFAULT1]]

; CHECK-SPIRV:      Label [[#DEFAULT1]]
; CHECK-SPIRV-NEXT: Switch [[#REGISTER]] [[#DEFAULT2:]] 0 [[#CASE3:]] 1 [[#CASE4:]]

; CHECK-SPIRV:      Label [[#CASE3]]
; CHECK-SPIRV-NEXT: Branch [[#DEFAULT2]]

; CHECK-SPIRV:      Label [[#CASE4]]
; CHECK-SPIRV-NEXT: Branch [[#DEFAULT2]]

; CHECK-SPIRV:      Label [[#DEFAULT2]]
; CHECK-SPIRV-NEXT: Return

; CHECK-LLVM: [[default1]]:
; CHECK-LLVM: switch i32 %[[value]], label %[[default2:[^ ]+]]
; CHECK-LLVM: [[default2]]:

}
