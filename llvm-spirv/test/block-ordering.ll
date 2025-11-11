; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s
; RUN: spirv-val %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; Checks SPIR-V blocks are correctly reordered so that dominators shows up
; before others in the binary layout.

define void @main() {
; CHECK: Label
; CHECK: Branch [[#l1:]]

; CHECK: Label [[#l1]]
; CHECK: Branch [[#l2:]]

; CHECK: Label [[#l2]]
; CHECK: Branch [[#end:]]

; CHECK: Label [[#end]]
; CHECK: Return
entry:
  br label %l1

l2:
  br label %end

l1:
  br label %l2

end:
  ret void
}
