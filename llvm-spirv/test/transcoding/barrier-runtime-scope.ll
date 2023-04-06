; This test checks that the translator is capable to correctly translate
; __spirv_ControlBarrier with runtime-known MemScope parameter
; to SPIR-V and back to OpenCL 2.0 IR.
; TODO: to remove this test once
; https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/1805
; is fixed as barrier.cl and sub_group_barrier.cl will be enough to test this
; case

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spv.txt
; RUN: FileCheck < %t.spv.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -to-binary %t.spv.txt -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv --spirv-target-env=CL2.0 -r %t.spv -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: FunctionCall [[#]] [[#SCOPE:]] [[#]]
; CHECK-SPIRV: ControlBarrier [[#]] [[#SCOPE]] [[#]]

; CHECK-LLVM: %[[#SCOPE:]] = call spir_func i32 @_Z8getScopev()
; CHECK-LLVM: [[CALL:%[a-z0-9]+]] = call spir_func i32 @__translate_spirv_memory_scope(i32 %[[#SCOPE]])
; CHECK-LLVM: call spir_func void @_Z17sub_group_barrierj12memory_scope(i32 3, i32 [[CALL]])

; CHECK-LLVM: define private spir_func i32 @__translate_spirv_memory_scope(i32 %key) {
; CHECK-LLVM: entry:
; CHECK-LLVM: switch i32 %key, label %default [
; CHECK-LLVM: i32 4, label %case.4
; CHECK-LLVM: i32 2, label %case.2
; CHECK-LLVM: i32 1, label %case.1
; CHECK-LLVM: i32 0, label %case.0
; CHECK-LLVM: i32 3, label %case.3
; CHECK-LLVM: ]
; CHECK-LLVM: default:                                          ; preds = %entry
; CHECK-LLVM: unreachable
; CHECK-LLVM: case.4:                                           ; preds = %entry
; CHECK-LLVM: ret i32 0
; CHECK-LLVM: case.2:                                           ; preds = %entry
; CHECK-LLVM: ret i32 1
; CHECK-LLVM: case.1:                                           ; preds = %entry
; CHECK-LLVM: ret i32 2
; CHECK-LLVM: case.0:                                           ; preds = %entry
; CHECK-LLVM: ret i32 3
; CHECK-LLVM: case.3:                                           ; preds = %entry
; CHECK-LLVM: ret i32 4
; CHECK-LLVM: }

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64-unknown-unknown"

define spir_func void @_Z3foov() {
  %1 = call noundef i32 @_Z8getScopev()
  call void @_Z22__spirv_ControlBarrieriii(i32 noundef 3, i32 noundef %1, i32 noundef 912)
  ret void
}

declare spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef, i32 noundef, i32 noundef)

declare spir_func i32 @_Z8getScopev()
