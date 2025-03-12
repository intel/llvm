; RUN: opt < %s -passes=msan -msan-instrumentation-with-call-threshold=0 -msan-eager-checks=1 -msan-spir-locals=1 -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

@WGLocal = internal addrspace(3) global i64 zeroinitializer, align 8

define spir_kernel void @MyKernel(ptr addrspace(3) noundef align 4 %_arg_acc) sanitize_memory {
; CHECK-LABEL: @MyKernel
entry:
  ; CHECK: %local_args = alloca i64, align 8
  ; CHECK-NEXT: %0 = getelementptr i64, ptr %local_args, i32 0
  ; CHECK-NEXT: %1 = ptrtoint ptr addrspace(3) %_arg_acc to i64
  ; CHECK-NEXT: store i64 %1, ptr %0, align 8
  ; CHECK-NEXT: %2 = ptrtoint ptr %local_args to i64
  ; CHECK-NEXT: call void @__msan_poison_shadow_dynamic_local(i64 %2, i32 1)

  ; CHECK: @__msan_poison_shadow_static_local{{.*}}@WGLocal
  ; CHECK: @__msan_barrier
  store i32 0, ptr addrspace(3) @WGLocal, align 8
  ; CHECK: @__msan_barrier
  ; CHECK: @__msan_unpoison_shadow_static_local{{.*}}@WGLocal

  ; CHECK: call void @__msan_unpoison_shadow_dynamic_local(i64 %2, i32 1)
  ret void
}
