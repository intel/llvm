; All OpVariable instructions in a function must be the first instructions in the first block

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-SPIRV: Function [[#]] [[#]] [[#]] [[#]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: Variable
; CHECK-SPIRV-NEXT: Variable
; CHECK-SPIRV: Return
; CHECK-SPIRV: FunctionEnd

define void @main() {
entry:
  %0 = alloca <2 x i32>, align 4
  %1 = getelementptr <2 x i32>, ptr %0, i32 0, i32 0
  %2 = alloca float, align 4
  ret void
}
