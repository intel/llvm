; RUN: opt < %s -passes=msan -msan-instrumentation-with-call-threshold=0 -msan-eager-checks=1 -msan-spir-privates=0 -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

; CHECK: @__MsanKernelMetadata = appending dso_local local_unnamed_addr addrspace(1) global
; CHECK-SAME: [[ATTR0:#[0-9]+]]

; CHECK-NOT: _tls

define spir_kernel void @MyKernel(ptr addrspace(1) noundef align 4 %_arg_array) sanitize_memory {
; CHECK-LABEL: define spir_kernel void @MyKernel
entry:
  %0 = load i32, ptr addrspace(1) %_arg_array, align 4
  ; CHECK:      %1 = ptrtoint ptr addrspace(1) %_arg_array to i64
  ; CHECK-NEXT: %2 = call ptr addrspace(1) @__msan_get_shadow(i64 %1, i32 1, ptr addrspace(2) null)
  ; CHECK-NEXT: %_msld = load i32, ptr addrspace(1) %2, align 4
  ; CHECK-NEXT: call void @__msan_maybe_warning_4(i32 zeroext %_msld, i64 zeroext 0, ptr addrspace(2) null, i32 0, ptr addrspace(2) @__msan_kernel)
  %call = call spir_func i32 @foo(i32 %0)
  ; CHECK:      %3 = ptrtoint ptr addrspace(1) %_arg_array to i64
  ; CHECK-NEXT: %4 = call ptr addrspace(1) @__msan_get_shadow(i64 %3, i32 1, ptr addrspace(2) null)
  ; CHECK-NEXT: store i32 0, ptr addrspace(1) %4, align 4
  store i32 %call, ptr addrspace(1) %_arg_array, align 4
  ret void
}

define spir_func i32 @foo(i32 %data) sanitize_memory {
; CHECK-LABEL: define spir_func i32 @foo
entry:
  ret i32 %data
}

; CHECK: attributes [[ATTR0]]
; CHECK-SAME: "sycl-device-global-size"="32" "sycl-device-image-scope" "sycl-host-access"="0" "sycl-unique-id"="_Z20__MsanKernelMetadata"
