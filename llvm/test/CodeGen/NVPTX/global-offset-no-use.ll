; RUN: opt -passes=globaloffset %s -S -o - | FileCheck %s

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; This test checks that unused __spirv_BuiltInGlobalOffset call is simplily
; deleted and there is no load from implicit parameter.

declare i64 @_Z27__spirv_BuiltInGlobalOffseti(i32)

define void @_ZTS14other_function() {
  %1 = tail call i64 @_Z27__spirv_BuiltInGlobalOffseti(i32 2)
  ret void
}

; CHECK-LABEL: define void @_ZTS14other_function() {
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }

; CHECK-LABEL: define void @_ZTS14other_function_with_offset(ptr %0) {
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }

define ptx_kernel void @_ZTS14example_kernel() {
entry:
  call void @_ZTS14other_function()
  ret void
}

!llvm.module.flags = !{!0}
!nvvm.annotations = !{!1, !2, !2, !1}

!0 = !{i32 1, !"sycl-device", i32 1}
!1 = distinct !{ptr @_ZTS14example_kernel, !"maxnreg", i32 256, !"kernel", i32 1}
!2 = !{ptr @_ZTS14example_kernel, !"maxntidx", i32 8, !"maxntidy", i32 16, !"maxntidz", i32 32}
