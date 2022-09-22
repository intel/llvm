; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK: 4 TypeInt [[INT:[0-9]+]] 32 0
; CHECK: 4 TypePointer [[INTPTR:[0-9]+]] 7 [[INT]]

; Function Attrs: nounwind
define spir_kernel void @foo() {
entry:
  %iptr = alloca i32, align 4
  %0 = call ptr @call(ptr %iptr)
  store i32 0, ptr %0, align 8
  ret void
}

define spir_func ptr @call(ptr %a) {
  ret ptr %a
}
