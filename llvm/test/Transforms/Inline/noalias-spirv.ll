; Test that alias scope metadata does not include function names for SPIR-V target.
; This reduces metadata bloat and improves compilation performance for SYCL.
;
; RUN: opt -passes=inline -enable-noalias-to-md-conversion -S < %s | FileCheck %s

; Check that optional string metadata node is not generated.
; CHECK-NOT: !"callee: %a"
; CHECK-NOT: !"callee2: %a"
; CHECK-NOT: !"callee2: %b"

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64-unknown-unknown"

define void @callee(ptr noalias nocapture %a, ptr nocapture readonly %c) #0 {
entry:
  %0 = load float, ptr %c, align 4
  %arrayidx = getelementptr inbounds float, ptr %a, i64 5
  store float %0, ptr %arrayidx, align 4
  ret void
}

; Don't check correctness precisely - just check if aliasing info is still
; generated during inlining for SPIR-V target.
; CHECK-LABEL: caller(
; CHECK: load float
; CHECK-SAME: !noalias ![[#Scope1:]]
; CHECK: store float
; CHECK-SAME: !alias.scope ![[#Scope1]]

define void @caller(ptr nocapture %a, ptr nocapture readonly %c) #0 {
entry:
  tail call void @callee(ptr %a, ptr %c)
  %0 = load float, ptr %c, align 4
  %arrayidx = getelementptr inbounds float, ptr %a, i64 7
  store float %0, ptr %arrayidx, align 4
  ret void
}

define void @callee2(ptr noalias nocapture %a, ptr noalias nocapture %b, ptr nocapture readonly %c) #0 {
entry:
  %0 = load float, ptr %c, align 4
  %arrayidx = getelementptr inbounds float, ptr %a, i64 5
  store float %0, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, ptr %b, i64 8
  store float %0, ptr %arrayidx1, align 4
  ret void
}

; Don't check correctness precisely - just check if aliasing info is still
; generated during inlining for SPIR-V target.
; CHECK-LABEL: caller2(
; CHECK: load float
; CHECK-SAME: !noalias ![[#]]
; CHECK: store float
; CHECK-SAME: !alias.scope ![[#Scope3:]], !noalias ![[#Scope4:]]
; CHECK: store float
; CHECK-SAME: !alias.scope ![[#Scope4:]], !noalias ![[#Scope3:]]

define void @caller2(ptr nocapture %a, ptr nocapture %b, ptr nocapture readonly %c) #0 {
entry:
  tail call void @callee2(ptr %a, ptr %b, ptr %c)
  %0 = load float, ptr %c, align 4
  %arrayidx = getelementptr inbounds float, ptr %a, i64 7
  store float %0, ptr %arrayidx, align 4
  ret void
}

attributes #0 = { nounwind }
