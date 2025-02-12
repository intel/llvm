; RUN: opt < %s -passes=msan -msan-instrumentation-with-call-threshold=0 -msan-eager-checks=1 -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

define spir_kernel void @MyKernelMemset(ptr %offset.i) {
; CHECK-LABEL: @MyKernelMemset
entry:
  call void @llvm.memset.p0.i64(ptr %offset.i, i8 0, i64 0, i1 false)
; CHECK: call ptr @__msan_memset_p0
  ret void
}

define spir_kernel void @MyKernelMemmove(ptr %x, ptr %y) {
; CHECK-LABEL: @MyKernelMemmove
entry:
  tail call void @llvm.memmove.p0.p0.i64(ptr %x, ptr %y, i64 0, i1 false)
; CHECK: call ptr @__msan_memmove_p0_p0
  ret void
}

define spir_kernel void @MyKernelMemcpy(ptr %x, ptr %y) {
; CHECK-LABEL: @MyKernelMemcpy
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr %x, ptr %y, i64 0, i1 false)
; CHECK: call ptr @__msan_memcpy_p0_p0
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #0

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1 immarg) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1 immarg) #1

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
