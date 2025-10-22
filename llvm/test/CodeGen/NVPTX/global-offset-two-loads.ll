; RUN: opt -bugpoint-enable-legacy-pm -globaloffset %s -S -o - | FileCheck %s

target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare ptr @llvm.nvvm.implicit.offset()

define weak_odr dso_local ptx_kernel void @test() {
entry:
; CHECK-NOT: call ptr @llvm.nvvm.implicit.offset()

  %0 = tail call ptr @llvm.nvvm.implicit.offset()
  %1 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %2 = load i32, ptr %1, align 4
  %3 = load i32, ptr %0, align 4
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"sycl-device", i32 1}
