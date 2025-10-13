; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

define i32 @test_switch_branches(i32 %a) {
entry:
  %alloc = alloca i32
; CHECK-SPIRV: Switch [[#]] [[#DEFAULT:]] 1 [[#CASE1:]] 2 [[#CASE2:]] 3 [[#CASE3:]]
; CHECK-LLVM: switch i32 %{{.*}}, label %[[default:]]
; CHECK-LLVM:   i32 1, label %[[case1:]]
; CHECK-LLVM:   i32 2, label %[[case2:]]
; CHECK-LLVM:   i32 3, label %[[case3:]]

  switch i32 %a, label %default [
    i32 1, label %case1
    i32 2, label %case2
    i32 3, label %case3
  ]

case1:
  store i32 1, ptr %alloc
  br label %end

case2:
  store i32 2, ptr %alloc
  br label %end

case3:
  store i32 3, ptr %alloc
  br label %end

default:
  store i32 0, ptr %alloc
  br label %end

end:
  %result = load i32, ptr %alloc
  ret i32 %result

; CHECK-SPIRV:      Label [[#CASE1]] 
; CHECK-SPIRV:      Branch [[#END:]]
; CHECK-SPIRV:      Label [[#CASE2]] 
; CHECK-SPIRV:      Branch [[#END]]
; CHECK-SPIRV:      Label [[#CASE3]]
; CHECK-SPIRV:      Branch [[#END:]]
; CHECK-SPIRV:      Label [[#DEFAULT]] 
; CHECK-SPIRV:      Branch [[#END]]
; CHECK-SPIRV:      Label [[#END]] 
; CHECK-SPIRV:      ReturnValue

; CHECK-LLVM: [[case1]]:
; CHECK-LLVM:   br label %[[end:]]
; CHECK-LLVM: [[case2]]:
; CHECK-LLVM:   br label %[[end]]
; CHECK-LLVM: [[case3]]:
; CHECK-LLVM:   br label %[[end]]
; CHECK-LLVM: [[default]]:
; CHECK-LLVM:   br label %[[end]]
; CHECK-LLVM: %[[end:]]
; CHECK-LLVM:   ret i32 %{{.*}}
}
