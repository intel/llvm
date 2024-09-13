; RUN: sycl-post-link -O0 --device-globals -ir-output-only -S %s -o %t.ll
; RUN: FileCheck %s -input-file=%t.ll

; The test checks that if sycl-post-link tool will skip handling non image scope
; device globals in SanitizerDeviceGlobal pass.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@dev_global = addrspace(1) global { ptr addrspace(1), [4 x i32] } zeroinitializer
@__DeviceSanitizerReportMem = addrspace(1) global { { i32, [257 x i8], [257 x i8], i32, i64, i64, i64, i64, i64, i64, i8, i32, i32, i32, i8 } } zeroinitializer

; CHECK: @dev_global = addrspace(1) global { ptr addrspace(1), [4 x i32] }
; CHECK-NOT: @__AsanDeviceGlobalCount
; CHECK-NOT: @__AsanDeviceGlobalMetadata
