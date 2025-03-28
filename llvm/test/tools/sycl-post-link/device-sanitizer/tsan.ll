; This test checks that the post-link tool properly generates "sanUsed=tsan"
; in [SYCL/misc properties], and fixes the attributes and metadata of @__TsanKernelMetadata

; RUN: sycl-post-link -properties -split=kernel -symbols -S < %s -o %t.table

; RUN: FileCheck %s -input-file=%t_0.prop --check-prefix CHECK-PROP
; CHECK-PROP: [SYCL/misc properties]
; CHECK-PROP: sanUsed=2|gAAAAAAAAAAdzFmb

; RUN: FileCheck %s -input-file=%t_0.ll --check-prefix CHECK-IR
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

@__TsanKernelMetadata = addrspace(1) global [1 x { i64, i64 }] [{ i64, i64 } { i64 0, i64 58 }]
; CHECK-IR: @__TsanKernelMetadata {{.*}} !spirv.Decorations
