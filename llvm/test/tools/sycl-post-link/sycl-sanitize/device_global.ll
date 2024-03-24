; RUN: sycl-post-link -O0 --device-globals -ir-output-only -S %s -o %t.ll
; RUN: FileCheck %s -input-file=%t.ll

; The test check that if sycl-post-link tool can correctly instrument non image
; scope device globals for sanitizer using. After SanitizeDeviceGlobal pass, non
; image scope device global should be padded with extra red zone. And two new
; globals will be added in the IR to record extra information.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@dev_global = addrspace(1) global { [4 x i32] } zeroinitializer #0
@__DeviceSanitizerReportMem = addrspace(1) global { { i32, [257 x i8], [257 x i8], i32, i64, i64, i64, i64, i64, i64, i8, i32, i32, i32, i8 } } zeroinitializer

; CHECK: @dev_global = addrspace(1) global { { [4 x i32] }, [16 x i8] }
; CHECK: @__AsanDeviceGlobalCount = local_unnamed_addr addrspace(1) global i64 1
; CHECK: @__AsanDeviceGlobalMetadata = local_unnamed_addr addrspace(1) global [1 x { i64, i64, i64 }] [{ i64, i64, i64 } { i64 16, i64 32, i64 ptrtoint (ptr addrspace(1) @dev_global to i64) }]

attributes #0 = { "sycl-device-global-size"="16" "sycl-device-image-scope" }
