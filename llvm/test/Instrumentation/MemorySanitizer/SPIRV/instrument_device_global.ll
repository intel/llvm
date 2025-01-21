; RUN: opt < %s -passes=msan -msan-instrumentation-with-call-threshold=0 -msan-eager-checks=1 -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

@.str = external addrspace(1) constant [59 x i8]
@__spirv_BuiltInGlobalInvocationId = external addrspace(1) constant <3 x i64>

; CHECK: @__MsanDeviceGlobalMetadata
; CHECK-NOT: @__spirv_BuiltInGlobalInvocationId
; CHECK-SAME: @.str
