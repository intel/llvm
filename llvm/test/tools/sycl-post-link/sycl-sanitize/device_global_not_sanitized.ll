; RUN: sycl-post-link -O0 --device-globals -ir-output-only -S %s -o %t.ll
; RUN: FileCheck %s -input-file=%t.ll

; The test checks that if sycl-post-link tool will skip running
; SanitizerDeviceGlobal pass if the IR is not sanitized

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@dev_global = addrspace(1) global { [4 x i32] } zeroinitializer

; CHECK: @dev_global = addrspace(1) global { [4 x i32] } zeroinitializer
; CHECK-NOT: @__AsanDeviceGlobalCount
; CHECK-NOT: @__AsanDeviceGlobalMetadata
