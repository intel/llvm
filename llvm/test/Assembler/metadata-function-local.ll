; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers | llvm-as | llvm-dis -opaque-pointers | FileCheck %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

declare void @llvm.metadata(metadata)

define void @foo(i32 %arg) {
entry:
  %before = alloca i32
  call void @llvm.metadata(metadata i32 %arg)
  call void @llvm.metadata(metadata ptr %after)
  call void @llvm.metadata(metadata ptr %before)
  %after = alloca i32
  ret void

; CHECK: %before = alloca i32
; CHECK: call void @llvm.metadata(metadata i32 %arg)
; CHECK: call void @llvm.metadata(metadata ptr %after)
; CHECK: call void @llvm.metadata(metadata ptr %before)
; CHECK: %after = alloca i32
}
