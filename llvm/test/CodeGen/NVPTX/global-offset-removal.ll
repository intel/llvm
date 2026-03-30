; RUN: opt -passes=globaloffset -enable-global-offset=false %s -S -o - | FileCheck %s

target triple = "nvptx64-nvidia-cuda"

; This test checks that __spirv_BuiltInGlobalOffset call is correctly removed

declare i64 @_Z27__spirv_BuiltInGlobalOffseti(i32)

define i64 @_ZTS14example_kernel() {
entry:
; CHECK-NOT: @call i64 @_Z27__spirv_BuiltInGlobalOffseti
; CHECK: ret i64 0
  %0 = tail call i64 @_Z27__spirv_BuiltInGlobalOffseti(i32 2)
  ret i64 %0
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"sycl-device", i32 1}
