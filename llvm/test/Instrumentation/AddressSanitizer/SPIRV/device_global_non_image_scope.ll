; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -asan-constructor-kind=none -S | FileCheck %s

; check non image scope device globals will not be instrumented.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@dev_global = addrspace(1) global { ptr addrspace(1), [4 x i32] } zeroinitializer

; CHECK: @dev_global = addrspace(1) global { ptr addrspace(1), [4 x i32] }
; CHECK-NOT: @__AsanDeviceGlobalMetadata
