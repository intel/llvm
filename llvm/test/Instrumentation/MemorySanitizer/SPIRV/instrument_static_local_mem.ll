; RUN: opt < %s -passes=msan -msan-instrumentation-with-call-threshold=0 -msan-eager-checks=1 -msan-spir-locals=1 -msan-spir-privates=0 -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

@WGCopy = internal addrspace(3) global i64 zeroinitializer, align 8
@WGLocal = internal addrspace(3) global i64 zeroinitializer, align 8
; CHECK: @__MsanKernelMetadata{{.*}}i64 14, i64 1, i64 0

define spir_kernel void @MyKernelMemset() sanitize_memory {
; CHECK-LABEL: @MyKernelMemset
entry:
  ; CHECK: @__msan_poison_shadow_static_local{{.*}}@WGLocal
  ; CHECK: @__msan_poison_shadow_static_local{{.*}}@WGCopy
  ; CHECK: @__msan_barrier
  store ptr addrspace(3) @WGLocal, ptr addrspace(3) @WGCopy, align 8
  ; CHECK: @__msan_barrier
  ; CHECK: @__msan_unpoison_shadow_static_local{{.*}}@WGCopy
  ; CHECK: @__msan_unpoison_shadow_static_local{{.*}}@WGLocal
  ret void
}
