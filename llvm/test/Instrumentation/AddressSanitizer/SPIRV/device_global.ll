; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -asan-constructor-kind=none -S | FileCheck %s

; check that image scope device globals can be correctly instrumented.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@dev_global = addrspace(1) global { [4 x i32] } zeroinitializer #0

; CHECK: @dev_global = addrspace(1) global { { [4 x i32] }, [16 x i8] }
; CHECK: @__AsanDeviceGlobalMetadata = appending local_unnamed_addr addrspace(1) global [1 x { i64, i64, i64 }] [{ i64, i64, i64 } { i64 16, i64 32, i64 ptrtoint (ptr addrspace(1) @dev_global to i64) }]

attributes #0 = { "sycl-device-global-size"="16" "sycl-device-image-scope" }
