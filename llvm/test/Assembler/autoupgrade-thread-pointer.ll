; Test autoupgrade of arch-specific thread pointer intrinsics
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers | FileCheck %s

declare ptr @llvm.aarch64.thread.pointer()
declare ptr @llvm.arm.thread.pointer()

define ptr @test1() {
; CHECK: test1()
; CHECK: call ptr @llvm.thread.pointer()
  %1 = call ptr @llvm.aarch64.thread.pointer()
  ret ptr %1
}

define ptr @test2() {
; CHECK: test2()
; CHECK: call ptr @llvm.thread.pointer()
  %1 = call ptr @llvm.arm.thread.pointer()
  ret ptr %1
}
