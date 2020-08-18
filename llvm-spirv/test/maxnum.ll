; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

define spir_func float @Test(float %x, float %y) {
entry:
  %0 = call float @llvm.maxnum.f32(float %x, float %y)
  ret float %0
}

; CHECK: Function
; CHECK: FunctionParameter {{[0-9]+}} [[x:[0-9]+]]
; CHECK: FunctionParameter {{[0-9]+}} [[y:[0-9]+]]
; CHECK: ExtInst {{[0-9]+}} [[res:[0-9]+]] {{[0-9]+}} fmax [[x]] [[y]]
; CHECK: ReturnValue [[res]]

declare float @llvm.maxnum.f32(float, float)
