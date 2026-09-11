; REQUIRES: spirv-dis
; RUN: llvm-spirv %s -o %t.spv
; RUN: spirv-val %t.spv
; RUN: spirv-dis %t.spv | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck --input-file=%t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: %[[Entry:[a-zA-Z0-9_]+]] = OpLabel
; CHECK-SPIRV: OpBranch %[[Label:[a-zA-Z0-9_]+]]
; CHECK-SPIRV: %[[Label]] = OpLabel
; CHECK-SPIRV: %{{.*}} = OpPhi %{{.*}} %{{.*}} %[[Entry]]

; CHECK-LLVM-LABEL: define spir_kernel void @f()
; CHECK-LLVM:       [[Entry:[a-zA-Z0-9_]+]]:
; CHECK-LLVM:       br label %[[Label:[a-zA-Z0-9_]+]]
; CHECK-LLVM:       [[Label]]:
; CHECK-LLVM:       %[[PHI:[a-zA-Z0-9_]+]] = phi i64 [ 0, %[[Entry]] ]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64-unknown-unknown"

define spir_kernel void @f() {
entry:
  br i1 undef, label %L815, label %L815

L815:                                             ; preds = %entry, %entry
  %v = phi i64 [ 0, %entry ], [ 0, %entry ]
  ret void
}
