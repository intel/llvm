; RUN: opt < %s -passes='function(tsan),module(tsan-module)' -tsan-instrument-func-entry-exit=0 -tsan-instrument-memintrinsics=0 -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

; CHECK-LABEL: @__TsanKernelMetadata = appending dso_local local_unnamed_addr addrspace(1) global
; CHECK-SAME: i64 ptrtoint (ptr addrspace(2) @__tsan_kernel to i64
; CHECK-SAME: [[ATTR0:#[0-9]+]]

; Function Attrs: sanitize_thread
define spir_kernel void @test() #0 {
entry:
  ret void
}

; CHECK: attributes [[ATTR0]] = {{.*}} "sycl-unique-id"="__TsanKernelMetadata098f6bcd4621d373cade4e832627b4f6"
attributes #0 = { sanitize_thread }
