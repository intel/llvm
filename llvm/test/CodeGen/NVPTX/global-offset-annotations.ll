; RUN: opt -passes=globaloffset %s -S -o - | FileCheck %s

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; This test checks that the transformation is applied to kernels found 

declare i64 @_Z27__spirv_BuiltInGlobalOffseti(i32)
; CHECK-NOT: _Z27__spirv_BuiltInGlobalOffseti

define i64 @_ZTS14other_function() {
  %1 = tail call i64 @_Z27__spirv_BuiltInGlobalOffseti(i32 2)
  ret i64 %1
}

define ptx_kernel void @_ZTS14example_kernel() {
entry:
  %0 = call i64 @_ZTS14other_function()
  ret void
}

!llvm.module.flags = !{!0}
!nvvm.annotations = !{!1, !2, !2, !1}

!0 = !{i32 1, !"sycl-device", i32 1}
!1 = distinct !{ptr @_ZTS14example_kernel, !"maxnreg", i32 256, !"kernel", i32 1}
!2 = !{ptr @_ZTS14example_kernel, !"maxntidx", i32 8, !"maxntidy", i32 16, !"maxntidz", i32 32}
