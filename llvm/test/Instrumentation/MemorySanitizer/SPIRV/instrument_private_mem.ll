; RUN: opt < %s -passes=msan -msan-instrumentation-with-call-threshold=0 -msan-eager-checks=1 -msan-spir-privates=1 -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

define spir_kernel void @MyKernel() sanitize_memory {
; CHECK-LABEL: @MyKernel
entry:
  %array = alloca [4 x i32], align 4
  ; CHECK: %0 = ptrtoint ptr %array to i64
  ; CHECK: %1 = call i64 @__msan_get_shadow(i64 %0, i32 0)
  ; CHECK: %2 = inttoptr i64 %1 to ptr addrspace(1)
  ; CHECK: call void @llvm.memset.p1.i64(ptr addrspace(1) align 4 %2, i8 -1, i64 16, i1 false)
  ret void
}
