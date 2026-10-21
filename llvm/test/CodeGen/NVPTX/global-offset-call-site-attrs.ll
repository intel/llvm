; RUN: opt -passes=globaloffset %s -S -o - | FileCheck %s

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; This test checks that call-site attributes (e.g. byval) are preserved on
; calls to the cloned `_with_offset` functions. NVPTX lowers byval arguments
; based on call-site attributes, so dropping them results in a mismatch
; between the call and the callee's formal parameters in the generated PTX.

%struct.S = type { [4 x i64] }

declare i64 @_Z27__spirv_BuiltInGlobalOffseti(i32)

define internal i64 @callee(ptr byval(%struct.S) align 8 %s) {
  %1 = tail call i64 @_Z27__spirv_BuiltInGlobalOffseti(i32 0)
  ret i64 %1
}

; CHECK: define internal i64 @callee_with_offset(ptr byval(%struct.S) align 8 %{{.*}}, ptr %{{.*}}) {
; CHECK: define internal i64 @caller_with_offset(ptr byval(%struct.S) align 8 %[[S:[0-9]+]], ptr %[[OFF:[0-9]+]]) {
; CHECK: call i64 @callee_with_offset(ptr byval(%struct.S) align 8 %[[S]], ptr %[[OFF]])
define internal i64 @caller(ptr byval(%struct.S) align 8 %s) {
  %1 = call i64 @callee(ptr byval(%struct.S) align 8 %s)
  ret i64 %1
}

; CHECK: define ptx_kernel void @kernel_with_offset(ptr byval([3 x i32]) %0) {
; CHECK: call i64 @caller_with_offset(ptr byval(%struct.S) align 8 %s, ptr %0)
define ptx_kernel void @kernel() {
entry:
  %s = alloca %struct.S, align 8
  %0 = call i64 @caller(ptr byval(%struct.S) align 8 %s)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"sycl-device", i32 1}
