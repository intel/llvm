; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK: 4 TypeInt [[INT:[0-9]+]] 32 0
; CHECK: 4 TypePointer [[INTPTR:[0-9]+]] 7 [[INT]]
; CHECK: 3 TypeFloat [[FLOAT:[0-9]+]] 32
; CHECK: 4 TypePointer [[FLOATPTR:[0-9]+]] 7 [[FLOAT]]

; Function Attrs: nounwind
define spir_kernel void @foo() {
; CHECK: 4 Variable [[INTPTR]] [[IPTR:[0-9]+]] 7
; CHECK: 4 Variable [[FLOATPTR]] [[FPTR:[0-9]+]] 7
; CHECK: Store [[IPTR]] {{[0-9]+}}
; CHECK: Load [[INT]] [[LOAD:[0-9]+]] [[IPTR]]
; CHECK: 4 Bitcast [[FLOAT]] [[BITCAST:[0-9]+]] [[LOAD]]
; CHECK: Store [[FPTR]] [[BITCAST]]
entry:
  %iptr = alloca i32, align 4
  %fptr = alloca float, align 4
  store i32 0, ptr %iptr, align 4
  %0 = load i32, ptr %iptr, align 4
  %1 = bitcast i32 %0 to float
  store float %1, ptr %fptr, align 4
  ret void
}
